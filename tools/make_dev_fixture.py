"""DEV ONLY: the evaluation samples published by Takehi et al., converted to the practice-baseline
schema.

Where it sits in the flow: nowhere in a real run. It gives Qwen3.5-0.8B a practice run to develop
and test against on a machine with no GPU baseline. These are EVALUATION-set problems, so a session
refuses this run's name unless smoketest=True. Never use it for a real row.

    python tools/make_dev_fixture.py --takehi-repo <clone of RikiyaT/lm_understand-logs>

Reads exp8_math/eval/unedited/parts/<set>_*/<set>_first_samples.jsonl.gz from the clone (sets gsm8k
and math500; plain data, nothing from the clone is executed) and the tiny model's tokenizer. Writes
data/practice_runs/<tiny_model>/think35/DEVFIXTURE_takehi_eval/{baseline.jsonl,summary.json}.

Those samples hold only id, answer and text, so three things here are stand-ins: the problem and
solution text are placeholders, a token count near the limit stands in for finish_reason, and
grading is plain string equality instead of math-verify. Good enough to exercise the code; the
numbers mean nothing.
"""
import argparse
import glob
import gzip
import json
import os

from tokenizers import Tokenizer

import run_practice_baseline as baseline

cfg = baseline.cfg

FIXTURE_NAME = cfg.DEV_FIXTURE_PREFIX + '_takehi_eval'
EVAL_SETS = {'gsm8k': 'gsm8k', 'math500': 'math'}  # the set name in that repository -> the source label used here
SAMPLES_GLOB = os.path.join('exp8_math', 'eval', 'unedited', 'parts', '{name}_*', '{name}_first_samples.jsonl.gz')
PLACEHOLDER = '[DEV FIXTURE: the published samples do not include this text]'
# Kept small on purpose: the full texts run to 100 KB each.
DEFAULT_ROWS_PER_SET = 16
CUT_TOKEN_MARGIN = 768  # re-encoding a decoded text does not return the exact count generated


def read_samples(takehi_repo, set_name, n_rows):
    pattern = os.path.join(takehi_repo, SAMPLES_GLOB.format(name=set_name))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No samples match {pattern}; pass --takehi-repo a clone of RikiyaT/lm_understand-logs.')
    samples = []
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            samples += [json.loads(line) for line in f]
    return samples[:n_rows]


def fixture_row(sample, source, tokenizer, max_tokens):
    text = sample['text']
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    finish_reason = baseline.FINISH_CUT if n_tokens >= max_tokens - CUT_TOKEN_MARGIN else baseline.FINISH_STOP
    extracted = baseline.extract_answer(text)
    problem = {'id': sample['id'], 'source': source, 'problem': PLACEHOLDER,
               'answer': sample['answer'], 'solution': PLACEHOLDER}
    return baseline.baseline_row(problem, text, n_tokens, finish_reason, extracted == sample['answer'], extracted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--takehi-repo', required=True)
    parser.add_argument('--rows-per-set', type=int, default=DEFAULT_ROWS_PER_SET)
    parser.add_argument('--tiny-model', default='Qwen3.5-0.8B')
    args = parser.parse_args()

    entry = cfg.tiny_entry(args.tiny_model)
    _, protocol_name = cfg.resolve_task(args.tiny_model, 'math')  # the samples are math answers
    protocol = cfg.PROTOCOLS[protocol_name]
    tokenizer = Tokenizer.from_file(os.path.join(entry['model_dir'], cfg.MODEL_TOKENIZER_FILE))
    rows = []
    for set_name, source in EVAL_SETS.items():
        samples = read_samples(args.takehi_repo, set_name, args.rows_per_set)
        rows += [fixture_row(s, source, tokenizer, protocol['sampling']['max_tokens']) for s in samples]

    out_dir = cfg.practice_run_dir(args.tiny_model, protocol_name, FIXTURE_NAME)
    summary = baseline.summarize(rows, task='math', protocol=protocol_name, sampling=protocol['sampling'],
                                 engine_version=None, tiny_model=args.tiny_model, model_revision=entry['revision'],
                                 dev_fixture=True)
    baseline.write_run(out_dir, rows, summary)
    size_mb = os.path.getsize(os.path.join(out_dir, cfg.BASELINE_FILE)) / 2**20
    print(f'{len(rows)} rows ({size_mb:.1f} MB) -> {out_dir}')
    print(json.dumps({key: summary[key] for key in ('n', 'accuracy', 'cut_share', 'failures')}))


if __name__ == '__main__':
    main()
