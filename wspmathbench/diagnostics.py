"""The diagnostic pack: the practice summary and K failing examples from a practice baseline run.

Where it sits in the flow: step 2 of the session. It gives the prompt its
measurements. It states what happened and never why: no failure mode is named here, because the
prompt is deliberately hint-free.

How an answer is shown depends on the task's style (cfg.TASKS): 'worked' answers are long, so the
first and last characters are shown next to the reference solution; 'short' answers are a few
tokens, so the whole continuation is shown.

Reads data/practice_runs/<tiny_model>/<protocol>/<practice_run>/{baseline.jsonl,summary.json} and the
task's practice files. Writes nothing.
"""
import json
import os
import random

from wspmathbench import config as cfg

_CORRECT = 'correct'


def build(tiny_model, practice_run, diag, task=None):
    """The pack for one session. diag is cfg.DIAG_DEFAULTS merged with the caller's settings;
    task=None means the tiny model's default task.

    Returns {'run_name', 'task', 'protocol', 'n_run', 'summary_text', 'examples_text',
    'example_ids', 'k', 'head_chars', 'tail_chars'}. Raises early if the run is missing or holds
    non-practice problems.
    """
    task, protocol = cfg.resolve_task(tiny_model, task)
    style = cfg.TASKS[task]['style']
    rows, summary = _read_run(cfg.practice_run_dir(tiny_model, protocol, practice_run))
    practice = _read_practice_details(task)
    if not practice_run.startswith(cfg.DEV_FIXTURE_PREFIX):
        _require_practice_problems(rows, practice, practice_run)

    failing = [row for row in rows if row['outcome'] != _CORRECT]
    k = min(diag['k'], len(failing))
    chosen = random.Random(diag['seed']).sample(failing, k)
    max_tokens = summary['sampling']['max_tokens']
    examples = [_render_example(i + 1, row, practice.get(row['id'], {}), diag, max_tokens, style)
                for i, row in enumerate(chosen)]
    return {'run_name': practice_run, 'task': task, 'protocol': protocol, 'n_run': len(rows),
            'summary_text': render_summary(summary, style), 'examples_text': '\n\n'.join(examples),
            'example_ids': [row['id'] for row in chosen], 'k': k,
            'head_chars': diag['head_chars'], 'tail_chars': diag['tail_chars']}


def render_summary(summary, style='worked'):
    """The measurements of a run. They stay in the prompt; nothing here says what to make of them."""
    by_source = ', '.join(f"{source} {_share(s['accuracy'])} ({s['n_correct']} of {s['n']})"
                          for source, s in summary['by_source'].items())
    failures = ', '.join(f'{outcome} {count}' for outcome, count in summary['failures'].items())
    lines = [f"Accuracy {_share(summary['accuracy'])} overall ({summary['n_correct']} of {summary['n']}); by source: {by_source}."]
    if style == 'worked':
        lines.append(f"Mean answer length {summary['mean_tokens']:,.0f} tokens. Answers cut at the token limit: {_share(summary['cut_share'])}.")
    lines.append(f'Failure counts: {failures}.')
    return '\n'.join(lines)


def _share(fraction):
    return f'{100 * fraction:.1f}%'


def _read_run(run_dir):
    baseline_path = os.path.join(run_dir, cfg.BASELINE_FILE)
    summary_path = os.path.join(run_dir, cfg.SUMMARY_FILE)
    if not (os.path.exists(baseline_path) and os.path.exists(summary_path)):
        raise FileNotFoundError(f'No practice run in {run_dir}. Create it with tools/run_practice_baseline.py '
                                f'(or, for a smoketest only, tools/make_dev_fixture.py).')
    with open(baseline_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    with open(summary_path, encoding='utf-8') as f:
        return rows, json.load(f)


def _read_practice_details(task):
    """id -> the practice problem's row, for the subject and level that baseline.jsonl leaves out."""
    details = {}
    for file_name in cfg.practice_files(task):
        with open(os.path.join(cfg.PRACTICE_DIR, file_name), encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                details[row['id']] = row
    return details


def _require_practice_problems(rows, practice, practice_run):
    # The pack must only ever be built from the task's practice problems.
    outside = [row['id'] for row in rows if row['id'] not in practice]
    if outside:
        raise ValueError(f"Practice run '{practice_run}' holds {len(outside)} problems that are not in the "
                         f"practice files, for example {outside[0]!r}. It cannot be used.")


def _render_example(number, row, details, diag, max_tokens, style):
    # Plain lines, no str.format: problems, solutions and answers are full of LaTeX braces.
    text = row['text']
    if style == 'short':
        # json.dumps shows the continuation's spaces and line breaks, which matter in a short answer.
        return '\n'.join([
            f'Example {number} ({_label(row, details)}). Outcome: {_short_outcome_sentence(row)}.',
            f"Question: {row['problem']}",
            f"Reference answer: {row['answer']}",
            f"Model's continuation: {json.dumps(text, ensure_ascii=False)}",
        ])
    return '\n'.join([
        f'Example {number} ({_label(row, details)}). Outcome: {_outcome_sentence(row, max_tokens)}.',
        f"Problem: {row['problem']}",
        f"Reference answer: {row['answer']}",
        f"Reference solution: {row['solution']}",
        f"Model's answer, first {diag['head_chars']} characters:",
        text[:diag['head_chars']],
        f"Model's answer, last {diag['tail_chars']} characters:",
        text[-diag['tail_chars']:],
    ])


def _label(row, details):
    parts = [row['source']]
    if 'subject' in details:
        parts.append(details['subject'])
    if 'level' in details:
        parts.append(details['level'].lower())
    return ', '.join(parts)


def _outcome_sentence(row, max_tokens):
    """What happened, nothing about why."""
    extracted, reference, outcome = row['extracted'], row['answer'], row['outcome']
    if outcome == 'wrong_answer':
        return f'wrong answer: {extracted}, reference {reference}'
    if outcome == 'no_answer_finished':
        return 'finished without a \\boxed{} answer'
    if extracted is None:
        return f'cut at {max_tokens:,} tokens, no \\boxed{{}} answer'
    return f'cut at {max_tokens:,} tokens, last \\boxed{{}} answer: {extracted}, reference {reference}'


def _short_outcome_sentence(row):
    if row['extracted'] is None:
        return 'no number in the continuation'
    return f"wrong answer: {row['extracted']}, reference {row['answer']}"
