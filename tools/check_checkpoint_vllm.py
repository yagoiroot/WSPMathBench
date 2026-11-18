"""Load a model folder in its engine and answer a few practice problems: a check that an edited
checkpoint still loads and still produces text. It is not a score.

Where it sits in the flow: after step 7 of a session, by hand. It runs on the Linux side (inside
WSL on Windows) in the vLLM venv, on the GPU. checkpoint.check_in_vllm starts it; by hand, in a
Linux shell at the repository root:

    ~/venvs/vllm/bin/python tools/check_checkpoint_vllm.py \
        --model-dir <models folder>/wsp_checkpoints/<session ID> --max-tokens 4096

An edited checkpoint's edits.json names its tiny model and task, so only the folder is needed. For
an unedited model's folder pass --tiny-model (default Qwen3.5-0.8B).

It loads the model exactly as run_practice_baseline.py does, with the same prompt format and
sampling (the evaluation protocol), except for a lower token limit by default so it finishes in
minutes. The same arguments give the same problems and sampling seeds, so running it on the base
model's folder gives a like-for-like comparison.

Reads the model folder and the practice files. Writes nothing.
"""
import argparse
import json
import time

import run_practice_baseline as baseline

cfg = baseline.cfg
TAIL_CHARS = 400
DEFAULT_N_PER_SOURCE = 2
DEFAULT_MAX_TOKENS = 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    baseline.add_model_arguments(parser)
    parser.set_defaults(n_gsm8k=DEFAULT_N_PER_SOURCE, n_math=DEFAULT_N_PER_SOURCE, n_per_source=None)
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help="per answer; never more than the protocol's own limit")
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error('--model-dir is required: an edited checkpoint folder, or the base model folder')

    record = cfg.read_checkpoint_record(args.model_dir)
    _print_edit_record(args.model_dir, record)
    if record and record.get('tiny_model'):
        args.tiny_model, args.task = record['tiny_model'], record.get('task')
    entry, task, protocol_name, model_dir, counts = baseline.resolve_model_arguments(args)
    if task != 'math' and args.n_per_source is None:
        counts = {source: DEFAULT_N_PER_SOURCE for source in counts}
    protocol = cfg.PROTOCOLS[protocol_name]
    max_tokens = min(args.max_tokens, protocol['sampling']['max_tokens'])
    protocol = {**protocol, 'sampling': {**protocol['sampling'], 'max_tokens': max_tokens}}

    problems = baseline.choose_problems(args.practice_dir, task, counts, args.seed)
    print(f'{args.tiny_model}: {len(problems)} practice problems, task {task}, protocol {protocol_name}, '
          f'at most {max_tokens:,} tokens each')
    started = time.time()
    outputs, engine_version = baseline.generate(model_dir, problems, protocol, entry, args.dtype)
    seconds = round(time.time() - started, 1)
    # Raises if any generation ended for a reason other than finishing or reaching the limit.
    grader = cfg.TASKS[task]['grader']
    rows = baseline.rows_from_outputs(problems, outputs, grader)

    for row in rows:
        print(f"\n--- {row['id']} ({row['source']}): {row['n_tokens']:,} tokens, finish {row['finish_reason']}, "
              f"answer {row['extracted']!r}, reference {row['answer']!r}, {row['outcome']}")
        print(f"... {row['text'][-TAIL_CHARS:]}")

    summary = baseline.summarize(rows, baseline.OUTCOMES[grader])
    print(f"\nLoaded and generated with {engine_version} in {seconds} s (loading included).")
    print(json.dumps({'model_dir': model_dir, 'n': summary['n'], 'n_correct': summary['n_correct'],
                      'mean_tokens': round(summary['mean_tokens']), 'cut_share': summary['cut_share'],
                      'max_tokens': max_tokens}))
    if all(not row['text'].strip() for row in rows):
        raise SystemExit('Every answer is empty: the model loads but does not produce text.')


def _print_edit_record(model_dir, record):
    if record is None:
        print(f'{model_dir}: no {cfg.CHECKPOINT_RECORD_FILE}, so not an edited checkpoint (the base model?)')
        return
    edits = ', '.join(f"{t['name']} ({t['entries_changed']:,} entries, relative change {t['relative_change']:.3f})"
                      for t in record['tensors'])
    print(f"{model_dir}: session {record.get('session_id')}, tiny model {record.get('tiny_model', 'not recorded')}, "
          f"edited: {edits}")


if __name__ == '__main__':
    main()
