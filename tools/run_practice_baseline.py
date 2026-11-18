"""The unedited tiny model answers practice problems with its evaluation protocol.

Where it sits in the flow: run once per tiny model by hand, before any session, on the Linux side
(inside WSL on Windows) in the vLLM venv with the GPU. diagnostics.py then reads its output, so the
answers the primary LLM sees were produced the same way the edited model will be scored. The runs
the paper's sessions used are in data/practice_runs/, so this is only needed for a new tiny model
or to reproduce them. In a Linux shell at the repository root:

    ~/venvs/vllm/bin/python tools/run_practice_baseline.py --tiny-model gemma-3-270m-it --run-name my_run

Everything else has a default taken from wspmathbench/config.py: the model folder (the registry's
model_dir), the task (the model's default task), the protocol (the model's protocol for that
task), the practice folder, the number of problems (128 from each source of the task; pass
--n-per-source 0 for the whole pool, as for TinyStories-33M's baseline_1000) and the output folder,
data/practice_runs/<tiny model>/<protocol>/<run name>/.

Reads the practice files and the model. Writes <out>/baseline.jsonl and <out>/summary.json, at the
very end and only if every generation finished properly. An existing run is not overwritten unless
--overwrite is given. Needs `pip install math-verify==0.9.0` in the vLLM venv for the math task
(the version Takehi et al. graded with).

The answer extraction and the checker call follow exp8_math/code/mathqa.py of Takehi et al. (MIT
licence).
vllm, transformers and math_verify are imported inside functions, so make_dev_fixture.py can reuse
the pure functions here on a machine that has none of them.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys

_PACKAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wspmathbench')


def _load_module(name):
    """A standard-library-only module of the package, loaded by its path: the vLLM venv does not
    have the package's other dependencies, so the package itself is not imported here."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PACKAGE_DIR, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfg = _load_module('config')
arithmetic = _load_module('arithmetic')

# The ways an answer can fail, per grader (cfg.TASKS[task]['grader']); 'correct' is the other outcome.
OUTCOMES = {'boxed': ('wrong_answer', 'no_answer_finished', 'cut_at_max_tokens'),
            'first_number': ('wrong_answer', 'no_number')}
FINISH_CUT = 'length'
FINISH_STOP = 'stop'
CHECKER_TIMEOUT_S = 5
DEFAULT_N_PER_SOURCE = 128


# ------------------------------------------------------------------------- answer extraction

def answer_text(text):
    """The part of a response that is graded: after the last </think>, or all of it if none."""
    return text.split('</think>')[-1].strip() if '</think>' in text else text


def last_boxed(text):
    """The content of the last \\boxed{...} (else the last \\fbox{...}), or None if there is none,
    it is empty or it is never closed (for example an answer cut inside the box)."""
    index = text.rfind('\\boxed')
    key = '\\boxed'
    if index < 0:
        index = text.rfind('\\fbox')
        key = '\\fbox'
        if index < 0:
            return None
    rest = text[index + len(key):]
    stripped = rest.lstrip()
    if stripped.startswith('{'):
        return _braced_content(stripped)
    if key == '\\boxed' and rest.startswith(' '):  # '\boxed 5': up to the next $ or line end
        return rest.strip().split('$')[0].split('\n')[0].strip() or None
    return None


def _braced_content(text):
    depth = 0
    for position, character in enumerate(text):
        depth += (character == '{') - (character == '}')
        if depth == 0:
            return text[1:position].strip() or None
    return None


def extract_answer(text):
    return last_boxed(answer_text(text))


def classify_outcome(correct, extracted, finish_reason):
    """One of 'correct' and OUTCOMES['boxed']. A correct box in an answer cut at the limit still counts."""
    if correct:
        return 'correct'
    if finish_reason == FINISH_CUT:
        return 'cut_at_max_tokens'
    return 'no_answer_finished' if extracted is None else 'wrong_answer'


def is_correct(extracted, reference):
    """math-verify equivalence of the boxed answer and the reference. GSM8K references are integer
    strings, so they are compared as numbers (an answer of 72.00 matches 72)."""
    if extracted is None or reference is None:
        return False
    import math_verify
    gold = math_verify.parse('\\boxed{' + reference + '}', parsing_timeout=CHECKER_TIMEOUT_S)
    prediction = math_verify.parse('\\boxed{' + extracted + '}', parsing_timeout=CHECKER_TIMEOUT_S)
    if not gold or not prediction:
        return False
    try:
        return bool(math_verify.verify(gold, prediction, timeout_seconds=CHECKER_TIMEOUT_S))
    except Exception:
        return False  # the checker gave up on this pair; the answer was not shown to be right


def grade(grader, text, reference, finish_reason):
    """(extracted answer, correct, outcome) of one response under a task's grader."""
    if grader == 'first_number':
        # A short completion nearly always ends at the token limit, so how it ended says nothing.
        extracted = arithmetic.first_number(text)
        correct = extracted is not None and extracted == int(reference)
        return extracted, correct, 'correct' if correct else 'no_number' if extracted is None else 'wrong_answer'
    extracted = extract_answer(text)
    correct = is_correct(extracted, reference)
    return extracted, correct, classify_outcome(correct, extracted, finish_reason)


# ---------------------------------------------------------------------------- problem choice

def choose_problems(practice_dir, task, counts, seed):
    """A seeded random sample of practice problems from each source of the task, in the task's
    source order. counts: source -> how many; a source with no count (or 0) gives all its rows."""
    rng = random.Random(seed)
    chosen = []
    for source, file_name in cfg.TASKS[task]['practice_sources'].items():
        with open(os.path.join(practice_dir, file_name), encoding='utf-8') as f:
            rows = [row for row in map(json.loads, f) if row['source'] == source]
        n = counts.get(source)
        chosen += rng.sample(rows, n) if n else rows
    return chosen


def stable_seed(problem_id):
    """A sampling seed fixed by the problem. Not hash(), which changes from process to process."""
    return int(hashlib.sha256(problem_id.encode('utf-8')).hexdigest()[:8], 16)


# ------------------------------------------------------------------------------------ output

def baseline_row(problem, text, n_tokens, finish_reason, correct, extracted, outcome=None):
    """One line of baseline.jsonl. outcome=None means the boxed grader's classification."""
    return {'id': problem['id'], 'source': problem['source'], 'problem': problem['problem'],
            'answer': problem['answer'], 'solution': problem.get('solution'), 'text': text,
            'n_tokens': n_tokens, 'finish_reason': finish_reason, 'extracted': extracted,
            'correct': correct, 'outcome': outcome or classify_outcome(correct, extracted, finish_reason)}


def summarize(rows, outcomes=OUTCOMES['boxed'], **details):
    """summary.json: counts, accuracy overall and by source, mean tokens, cut share, failure counts.
    details (the sampling used, the engine version, the model revision, ...) are stored as given."""
    sources = sorted({row['source'] for row in rows})
    return {**_counts(rows),
            'by_source': {source: _counts([r for r in rows if r['source'] == source]) for source in sources},
            'failures': {outcome: sum(r['outcome'] == outcome for r in rows) for outcome in outcomes},
            **details}


def _counts(rows):
    n = len(rows)
    n_correct = sum(bool(r['correct']) for r in rows)
    return {'n': n, 'n_correct': n_correct, 'accuracy': n_correct / n,
            'mean_tokens': sum(r['n_tokens'] for r in rows) / n,
            'cut_share': sum(r['finish_reason'] == FINISH_CUT for r in rows) / n}


def write_run(out_dir, rows, summary):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, cfg.BASELINE_FILE), 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(os.path.join(out_dir, cfg.SUMMARY_FILE), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)


# -------------------------------------------------------------------------------- generation

def generate(model_dir, problems, protocol, entry, dtype=None):
    """One generation per problem with the tiny model's engine.
    Returns ([(text, number of new tokens, finish reason), ...], engine version)."""
    stop_ids = stop_token_ids(model_dir, entry)
    if entry.get('engine', 'vllm') == 'transformers':
        import transformers_engine
        model, tokenizer, version = transformers_engine.load(model_dir, dtype or cfg.BASELINE_VLLM['dtype'])
        return [transformers_engine.generate(model, tokenizer, render_prompt(tokenizer, problem, protocol),
                                             protocol['sampling'], stop_ids, stable_seed(problem['id']))
                for problem in problems], version

    os.environ.update(cfg.BASELINE_ENV)  # read when vllm is imported
    # FlashInfer compiles its sampling kernel on first use and calls `ninja` by name. ninja lives in
    # the venv's bin/, which is on PATH only if the venv was activated, not when its python is
    # called directly as in the command above.
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    load_arguments = {**cfg.BASELINE_VLLM, 'max_model_len': entry['max_model_len'], **entry['vllm_kwargs']}
    if dtype:
        load_arguments['dtype'] = dtype
    llm = vllm.LLM(model=model_dir, **load_arguments)
    # Tokenised here with no special tokens added: a chat template writes its own (Gemma's and
    # MobileLLM-R1's begin with the start-of-text token), and vLLM would add a second one to a
    # text prompt. Qwen's tokenizer adds none either way, so its prompts are what they always were.
    prompts = [{'prompt_token_ids': tokenizer(render_prompt(tokenizer, problem, protocol),
                                              add_special_tokens=False).input_ids} for problem in problems]
    params = [vllm.SamplingParams(seed=stable_seed(problem['id']), stop_token_ids=stop_ids, **protocol['sampling'])
              for problem in problems]
    outputs = llm.generate(prompts, params)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outputs], \
        f'vllm {vllm.__version__}'


def render_prompt(tokenizer, problem, protocol):
    """The text the model is given for one problem, by the protocol's format."""
    if protocol['format'] == 'completion':
        return problem['problem']
    role = protocol['instruction_role']
    messages = [{'role': 'system', 'content': cfg.BOXED_INSTRUCTION}] if role == 'system' else []
    user_message = problem['problem'] + ('\n' + cfg.BOXED_INSTRUCTION if role == 'user' else '')
    messages.append({'role': 'user', 'content': user_message})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **protocol['template_kwargs'])


stop_token_ids = cfg.stop_token_ids  # shared with the scoring server (llm/query.py), which sends the same ids


def rows_from_outputs(problems, outputs, grader='boxed'):
    rows = []
    for problem, (text, n_tokens, finish_reason) in zip(problems, outputs):
        if finish_reason not in (FINISH_STOP, FINISH_CUT):
            # An engine stopped from outside returns partial answers that look like real ones.
            raise RuntimeError(f"{problem['id']} ended with finish_reason={finish_reason!r}; nothing was saved.")
        extracted, correct, outcome = grade(grader, text, problem['answer'], finish_reason)
        rows.append(baseline_row(problem, text, n_tokens, finish_reason, correct, extracted, outcome))
    return rows


def add_model_arguments(parser):
    """The arguments this tool and check_checkpoint_vllm.py share."""
    parser.add_argument('--tiny-model', default='Qwen3.5-0.8B', help='a key of the registry in config.py')
    parser.add_argument('--task', default=None, help="default: the tiny model's default task")
    parser.add_argument('--model-dir', default=None, help="default: the registry's model_dir")
    parser.add_argument('--practice-dir', default=cfg.PRACTICE_DIR)
    parser.add_argument('--n-gsm8k', type=int, default=DEFAULT_N_PER_SOURCE, help='math task: problems from GSM8K train')
    parser.add_argument('--n-math', type=int, default=DEFAULT_N_PER_SOURCE, help='math task: problems from MATH train')
    parser.add_argument('--n-per-source', type=int, default=None,
                        help=f'problems from each source of the task, overriding the two above; 0 means all '
                             f'(default for other tasks: {DEFAULT_N_PER_SOURCE})')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', default=None, help="default: bfloat16; float32 is for the one-off check "
                                                      "against a model's published float32 weights")


def resolve_model_arguments(args):
    """(registry entry, task, protocol name, model folder, counts per source) from the shared arguments."""
    entry = cfg.tiny_entry(args.tiny_model)
    task, protocol_name = cfg.resolve_task(args.tiny_model, args.task)
    sources = list(cfg.TASKS[task]['practice_sources'])
    if args.n_per_source is not None:
        counts = {source: args.n_per_source for source in sources}
    elif task == 'math':
        counts = {'gsm8k': args.n_gsm8k, 'math': args.n_math}
    else:
        counts = {source: DEFAULT_N_PER_SOURCE for source in sources}
    return entry, task, protocol_name, args.model_dir or cfg.to_wsl_path(entry['model_dir']), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument('--run-name', default='baseline_256', help='the folder name under data/practice_runs/<tiny model>/<protocol>/')
    parser.add_argument('--out', default=None, help='the whole output folder, instead of the default built from --run-name')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing run')
    args = parser.parse_args()

    entry, task, protocol_name, model_dir, counts = resolve_model_arguments(args)
    protocol = cfg.PROTOCOLS[protocol_name]
    out_dir = args.out or cfg.practice_run_dir(args.tiny_model, protocol_name, args.run_name)
    if os.path.exists(os.path.join(out_dir, cfg.BASELINE_FILE)) and not args.overwrite:
        sys.exit(f'{out_dir} already holds a run. Pick another --run-name, or pass --overwrite to replace it.')
    problems = choose_problems(args.practice_dir, task, counts, args.seed)
    print(f'{args.tiny_model}: {len(problems)} problems, task {task}, protocol {protocol_name}, model {model_dir}, out {out_dir}')

    outputs, engine_version = generate(model_dir, problems, protocol, entry, args.dtype)
    grader = cfg.TASKS[task]['grader']
    rows = rows_from_outputs(problems, outputs, grader)
    summary = summarize(rows, OUTCOMES[grader], task=task, protocol=protocol_name, sampling=protocol['sampling'],
                        seed=args.seed, engine_version=engine_version, tiny_model=args.tiny_model,
                        model_revision=entry['revision'], model_dir=model_dir,
                        dtype=args.dtype or cfg.BASELINE_VLLM['dtype'])
    write_run(out_dir, rows, summary)
    print(json.dumps({key: summary[key] for key in ('n', 'accuracy', 'by_source', 'mean_tokens', 'cut_share', 'failures')}))


if __name__ == '__main__':
    main()
