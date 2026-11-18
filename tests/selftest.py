"""End-to-end check of the benchmark with the stub provider. No API key, no GPU, no spend.

Run it from the repository root after the one-time setup (docs/SETUP.md) and after any change:

    python tests/selftest.py [--practice-run NAME] [--build-checkpoint] [--skip-kernel-limits]
                             [--only-model KEY]

What needs no tiny model is always checked: reply parsing, the prompt templates, BasicArithmetic
and the pure functions of the baseline tool. Each registered tiny model whose folder exists under
cfg.MODELS_DIR is then checked; a model whose folder is missing is reported as NOT CHECKED and
does not count as a failure.

Qwen3.5-0.8B gets the full set of checks, with a real practice run (by default the dev fixture if
it exists, otherwise baseline_256). Every other tiny model gets the checks that depend on the
architecture (edit rules, edit language, prompts for both versions and tiers, a stub session per
tier) against a made-up practice run in a temporary folder. For every model, the prompts rendered
from its shipped practice run are compared with the reference copies in tests/reference_prompts/.
--only-model limits the run to one tiny model.

The stub-code sessions and the kernel-limit checks start the oneshot_code kernel, which needs the
sandbox venv on the Linux side (WSL on Windows); --skip-kernel-limits leaves out all of them except
Qwen's stub-code sessions. Nothing here starts vLLM or uses the GPU.

Uses <results>/smoketest/, which it clears before each run and at the end, and a temporary folder
for the master-row test, so the real master CSV is never touched. --build-checkpoint also writes
Qwen's 1.7 GB checkpoint to cfg.SMOKETEST_CHECKPOINT_DIR, and removes it (the smallest model's
checkpoint, 0.13 GB, is always built and removed).

Exits with status 1 if any check fails.
"""
import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root

import polars as pl  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from wspmathbench import config as cfg  # noqa: E402
from wspmathbench import arithmetic, benchmark, diagnostics, dsl, edits, kernel, kernel_runner  # noqa: E402
from wspmathbench import prompts, session, stub  # noqa: E402
from wspmathbench import tiny_model as tiny_model_module  # noqa: E402
from wspmathbench.llm import registry  # noqa: E402

TINY_MODEL = 'Qwen3.5-0.8B'
SMALLEST_MODEL = 'TinyStories-33M'
FIXTURE_RUN = cfg.DEV_FIXTURE_PREFIX + '_takehi_eval'
SYNTHETIC_RUN = 'selftest_run'
REFERENCE_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_prompts')
# Words that would name a failure mode. The templates, the edit language and the task and
# architecture sentences must not use them. A protocol's sentence is not checked: it has to state
# the decoding settings, and LFM2.5's include a repetition penalty.
HINT_WORDS = ('repetition', 'repeat', 'loop', 'stuck', 'wander', 'wait')
FAKE_PATHS = {'model_dir': '/mnt/c/MODEL', 'practice_dir': '/mnt/e/PRACTICE', 'work_dir': '/mnt/e/WORK'}

_failures = []
_not_checked = []


def check(label, condition, detail=''):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}{'' if condition else f'  -> {detail}'}")
    if not condition:
        _failures.append(label)


def section(title):
    print(f'\n=== {title}')


def refusal_of(function):
    """The message of the EditRefused that `function` raises, or None if it raises nothing."""
    try:
        function()
    except edits.EditRefused as exc:
        return str(exc)
    return None


def hint_words_in(text):
    return [word for word in HINT_WORDS if word in text.lower()]


# ------------------------------------------------------------------------------- the edit rules

def check_edit_rules(tiny):
    section(f'{tiny.key}: edit rules')
    target, _ = stub.targets(tiny)
    original = tiny.load(target)
    with tempfile.TemporaryDirectory() as edited_dir:
        def save(tensor, name=target, reference=original):
            return edits.save_edit(edited_dir, name, tensor, reference, tiny.names, tiny.frozen)

        final_norm = tiny.entry['final_norm']
        norm_values = tiny.load(final_norm)
        refused = refusal_of(lambda: save(norm_values * 2, final_norm, norm_values))
        check('frozen final norm refused', refused and 'frozen' in refused, refused)
        refused = refusal_of(lambda: save(original, 'lm_head.weight', None))
        check('frozen lm_head refused', refused and 'frozen' in refused, refused)
        check('every frozen tensor of the file is readable but not editable',
              all(name not in tiny.names for name in tiny.frozen)
              and all(name in tiny.language_names for name in tiny.frozen if name != 'lm_head.weight'))
        refused = refusal_of(lambda: save(original, 'model.visual.merger.linear_fc1.weight', None))
        check('a tensor outside the language model refused', refused and 'not an editable' in refused, refused)
        refused = refusal_of(lambda: save(original * 0.5))
        check('pure rescale refused', refused and 'constant' in refused, refused)
        refused = refusal_of(lambda: save(original * 0.0))
        check('zeroing a whole tensor refused', refused and 'constant' in refused, refused)
        refused = refusal_of(lambda: save(original[:10]))
        check('wrong shape refused', refused and 'shape' in refused, refused)
        with_nan = original.clone()
        with_nan[0, 0] = float('nan')
        refused = refusal_of(lambda: save(with_nan))
        check('non-finite value refused', refused and 'finite' in refused, refused)

        check('no edit is dropped silently', save(original.clone()) is False and edits.edited(edited_dir) == [])
        edit = original.clone()
        edit[:, :16] *= 0.5
        check('a real edit is saved', save(edit) is True and edits.edited(edited_dir) == [target])
        manifest = edits.manifest(edited_dir, tiny)
        check('manifest counts the changed entries', manifest['entries_changed'] == tiny.shapes[target][0] * 16,
              manifest['entries_changed'])
        check('restoring the original drops the edit', save(original.clone()) is False and edits.edited(edited_dir) == [])
        save(edit)
        edits.drop_edit(edited_dir, target)
        check('drop_edit undoes a save', edits.edited(edited_dir) == [])


# ---------------------------------------------------------------------------- the edit language

def whole_tensor_rescale_ops(tiny):
    """Valid operations that together only rescale one whole tensor, or None if the language's
    limits leave no way to write that for this model."""
    for name in tiny.names:
        if tiny.shapes[name][0] <= cfg.MAX_INDICES:
            return [{'op': 'scale_rows', 'tensor': name, 'rows': list(range(tiny.shapes[name][0])), 'factor': 0.5}]
    if tiny.num_attention_heads < cfg.MAX_OPS:
        layer = tiny.head_layers()[0]
        return [{'op': 'zero_head', 'layer': layer, 'head': head} for head in range(tiny.num_attention_heads)]
    return None


def check_dsl(tiny):
    section(f'{tiny.key}: edit language validation')
    first, second = stub.targets(tiny)
    column, missing = stub.out_of_range_column(tiny), stub.missing_tensor(tiny)
    headless = [layer for layer in range(len(tiny.layer_types)) if not tiny.has_heads(layer)]
    several = 'Wait a moment'
    cases = [
        ('out-of-range col', {'op': 'zero_columns', 'tensor': first, 'cols': [column]},
         f'edits[0]: col {column} is out of range for shape {tiny.shapes[first]}'),
        ('unknown tensor', {'op': 'zero_rows', 'tensor': missing, 'rows': [0]},
         f"edits[0]: tensor '{missing}' is not an editable tensor"),
        ('frozen tensor', {'op': 'zero_rows', 'tensor': tiny.entry['embedding'], 'rows': [0]},
         f"edits[0]: tensor '{tiny.entry['embedding']}' is frozen and cannot be edited"),
        ('head on a layer without heads', {'op': 'zero_head', 'layer': headless[0], 'head': 0},
         f'edits[0]: layer {headless[0]} is not a full-attention layer') if headless else
        ('head on a layer that does not exist', {'op': 'zero_head', 'layer': len(tiny.layer_types), 'head': 0},
         f'edits[0]: layer {len(tiny.layer_types)} does not exist: the layers are 0 to {len(tiny.layer_types) - 1}'),
        ('head out of range', {'op': 'zero_head', 'layer': tiny.head_layers()[0], 'head': tiny.num_attention_heads},
         f'edits[0]: head {tiny.num_attention_heads} is out of range: layer {tiny.head_layers()[0]} has heads 0 to '
         f'{tiny.num_attention_heads - 1}'),
        ('factor too large', {'op': 'scale_rows', 'tensor': first, 'rows': [0], 'factor': 5},
         "edits[0]: 'factor' must be a number between 0 and 4, got 5"),
        ('token of several ids', {'op': 'project_out', 'tensor': first, 'strength': 1, 'cols': 'all',
                                  'direction': {'type': 'unembed_token', 'token': several}},
         f"edits[0]: token '{several}' encodes to {len(tiny.encode(several))} token ids, it must encode to exactly one"),
    ]
    for label, op, expected in cases:
        errors = dsl.validate({'edits': [op]}, tiny)
        check(label, errors == [expected], errors)
    too_many = {'edits': [{'op': 'zero_rows', 'tensor': first, 'rows': [0]}] * (cfg.MAX_OPS + 1)}
    check('too many operations', len(dsl.validate(too_many, tiny)) == 1)
    two_problems = {'edits': [cases[0][1], cases[1][1]]}
    check('every problem is reported', len(dsl.validate(two_problems, tiny)) == 2)
    check('bad JSON is a validation error', 'not valid JSON' in dsl.parse('{"edits": [')[1][0])
    check('an empty edit list is valid', dsl.validate({'edits': []}, tiny) == [])
    check("the language's example tokens are single tokens here",
          all(len(tiny.encode(token)) == 1 for token in stub.TOKENS), [tiny.encode(t) for t in stub.TOKENS])
    head_columns = tiny.shapes[tiny.attention_output_name(tiny.head_layers()[0])][1]
    check('heads divide the attention output evenly', tiny.head_dim * tiny.num_attention_heads == head_columns,
          (tiny.head_dim, tiny.num_attention_heads, head_columns))

    direction = {'type': 'unembed_token', 'token': 'the', 'fold_final_norm': True}
    post_norm_op = {'op': 'project_out', 'tensor': first, 'strength': 1, 'cols': 'all',
                    'direction': {**direction, 'fold_post_norm': False}}
    if tiny.entry['post_norms']:
        check('fold_post_norm is accepted', dsl.validate({'edits': [post_norm_op]}, tiny) == [])
        folded, source = dsl.resolve_direction(direction, tiny, target=first)
        plain, _ = dsl.resolve_direction({**direction, 'fold_post_norm': False}, tiny, target=first)
        gain = 1.0 + tiny.load(tiny.post_norm_name(first))
        expected = plain * gain / (plain * gain).norm()
        check('by default a direction is folded through the norm after the tensor',
              source['fold_post_norm'] and source['post_norm'] == tiny.post_norm_name(first)
              and torch.allclose(folded, expected, atol=1e-5), source)
    else:
        errors = dsl.validate({'edits': [post_norm_op]}, tiny)
        check('fold_post_norm is refused for a model without post-block norms',
              errors == ["edits[0]: direction 'unembed_token' has an unknown field 'fold_post_norm'"], errors)

    biases = [name for name in tiny.names if tiny.is_residual_bias(name)]
    if biases:
        bias_op = {'op': 'project_out', 'tensor': biases[0], 'strength': 1, 'cols': 'all', 'direction': direction}
        check('project_out accepts a residual bias', dsl.validate({'edits': [bias_op]}, tiny) == [])
        errors = dsl.validate({'edits': [{**bias_op, 'cols': [0]}]}, tiny)
        check('project_out on a bias needs cols "all"', len(errors) == 1 and '"all"' in errors[0], errors)
        with tempfile.TemporaryDirectory() as edited_dir:
            record, errors = dsl.apply({'edits': [bias_op]}, tiny, edited_dir)
            u, _ = dsl.resolve_direction(direction, tiny, target=biases[0])
            edited = torch.from_numpy(edits.load_edit(edited_dir, biases[0])) if not errors else None
            check('the direction is gone from the edited bias',
                  not errors and abs(float(u @ edited)) < 1e-4 * float(edited.norm()), errors)

    rescale = whole_tensor_rescale_ops(tiny)
    if rescale is None:
        print('  (no way to write a whole-tensor rescale within the limits for this model; not checked)')
    else:
        with tempfile.TemporaryDirectory() as edited_dir:
            spec = {'edits': [{'op': 'zero_rows', 'tensor': second, 'rows': [0]}] + rescale}
            check('the rescale spec is valid', dsl.validate(spec, tiny) == [], dsl.validate(spec, tiny))
            _, errors = dsl.apply(spec, tiny, edited_dir)
            check('a refusal while applying keeps nothing', errors and edits.edited(edited_dir) == [], errors)


def check_block_parsing():
    section('reply parsing')
    parse = session.parse_single_block
    check('one block', parse('text\n```json\n{}\n```\nafter', 'json') == ('{}', None))
    check('no block', parse('only prose', 'json')[0] is None)
    check('two blocks', parse('```json\n{}\n```\n```json\n[]\n```', 'json')[0] is None)
    check('unclosed fence is no block', parse('```python\nx = 1', 'python')[0] is None)
    check('other fences do not count', parse('```text\nnote\n```\n```python\nx = 1\n```', 'python') == ('x = 1', None))


def check_templates():
    section('prompt templates and task sentences')
    for version in cfg.PROMPT_VERSIONS:
        for tier in cfg.TIERS:
            with open(os.path.join(cfg.PROMPTS_DIR, f'{tier}_{version}.txt'), encoding='utf-8') as f:
                found = hint_words_in(f.read())
            check(f'{tier}_{version} template has no failure-mode words', not found, found)
        check(f'system_{version} exists', bool(prompts.system_prompt(version)))
    for task, entry in cfg.TASKS.items():
        found = hint_words_in(' '.join(value for key, value in entry.items() if key.endswith('_text')))
        check(f"task '{task}' sentences have no failure-mode words", not found, found)
    check('retry suffix carries the stub marker', stub.RETRY_MARKER in prompts.RETRY_SUFFIX)


def check_prompts(tiny, runs_dir):
    """Both versions and both tiers render for this model, from a made-up practice run."""
    section(f'{tiny.key}: prompts')
    found = hint_words_in(dsl.render_dsl_text(tiny))
    check('edit language has no failure-mode words', not found, found)
    found = hint_words_in(tiny.entry['arch_text'])
    check('architecture sentence has no failure-mode words', not found, found)
    real_runs = cfg.PRACTICE_RUNS_DIR
    cfg.PRACTICE_RUNS_DIR = runs_dir
    try:
        pack = diagnostics.build(tiny.key, SYNTHETIC_RUN, dict(cfg.DIAG_DEFAULTS))
    finally:
        cfg.PRACTICE_RUNS_DIR = real_runs
    task = cfg.TASKS[pack['task']]
    for version in cfg.PROMPT_VERSIONS:
        for tier in cfg.TIERS:
            prompt = prompts.render(tier, tiny, pack, FAKE_PATHS, version)
            check(f'{tier} {version} renders with the model, task and protocol',
                  tiny.key in prompt and task['evaluation_text'] in prompt and tiny.names[0] in prompt
                  and cfg.PROTOCOLS[pack['protocol']]['text'] in prompt and '{' + 'goal_text' + '}' not in prompt)
    for name in tiny.residual_writers()[:1]:
        check('the tensor table and the edit language agree on the residual width',
              f'shape [{tiny.shapes[name][0]}, k]' in dsl.render_dsl_text(tiny))


def check_reference_prompts(tiny):
    """The prompt rendered from the model's shipped practice run must stay what the reference copy
    says, character for character: the prompt is part of the treatment. The reference copies were
    rendered with FAKE_PATHS in place of a machine's folders."""
    section(f'{tiny.key}: prompts against the reference copies')
    practice_run = benchmark.MODEL_PRACTICE_RUNS[tiny.key]
    _, protocol = cfg.resolve_task(tiny.key)
    version = cfg.resolve_version(tiny.key)
    references = {tier: os.path.join(REFERENCE_PROMPTS_DIR, f'{tiny.key}_{tier}_{version}.txt') for tier in cfg.TIERS}
    if not (os.path.isdir(cfg.practice_run_dir(tiny.key, protocol, practice_run))
            and all(os.path.exists(path) for path in references.values())):
        print(f'  ({practice_run} or the reference copies are not on this machine; not checked)')
        return
    pack = diagnostics.build(tiny.key, practice_run, dict(cfg.DIAG_DEFAULTS))
    for tier, path in references.items():
        with open(path, encoding='utf-8', newline='') as f:
            reference = f.read()
        rendered = prompts.render(tier, tiny, pack, FAKE_PATHS, version)
        check(f'the {tier} {version} prompt is identical to its reference copy', rendered == reference,
              f'{len(rendered)} characters against {len(reference)}')


def check_arithmetic():
    section('BasicArithmetic')
    cases = {' 4': 4, ' four.': 4, 'twenty-three apples': 23, 'twenty three': 23, 'twenty. Three': 20,
             'one hundred and five': 105, '-3': -3, '5-2': 5, 'The answer is 12.5': 12, 'no idea': None, '': None,
             'four five': 4, 'ninety-nine': 99, 'zero': 0, ' 4\n3 + 3 = 6': 4}
    wrong = {text: arithmetic.first_number(text) for text, value in cases.items()
             if arithmetic.first_number(text) != value}
    check('the first number of a continuation is read', not wrong, wrong)
    check('both forms of a question', arithmetic.question('add', 2, 2, 'symbolic') == '2 + 2 ='
          and arithmetic.question('mul', 23, 4, 'word') == 'twenty-three times four equals'
          and arithmetic.question('div', 56, 7, 'symbolic') == '56 / 7 ='
          and arithmetic.question('mul', 1, 2, 'symbolic', 'x', spaced=False) == '1x2='
          and arithmetic.question('mul', 1, 2, 'symbolic', '*') == '1 * 2 ='
          and arithmetic.question('div', 56, 7, 'word') == 'fifty-six divided by seven equals')
    try:
        arithmetic.answer('div', 7, 2)
        check('a division that does not come out whole is refused', False)
    except ValueError:
        check('a division that does not come out whole is refused', arithmetic.answer('div', 56, 7) == 8)
    practice_path = os.path.join(cfg.PRACTICE_DIR, cfg.practice_files('basic_arithmetic')[0])
    with open(practice_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    regraded = [row for row in rows if arithmetic.first_number(' ' + row['answer']) != int(row['answer'])
                or arithmetic.question(row['operation'], row['a'], row['b'], row['source'], row['symbol'],
                                       row['spacing'] != 'tight') != row['problem']
                or arithmetic.answer(row['operation'], row['a'], row['b']) != int(row['answer'])]
    check('every practice question matches its numbers and its answer', len(rows) == 1000 and not regraded, regraded[:2])
    symbolic = [row for row in rows if row['source'] == 'symbolic']
    tight = sum(row['spacing'] == 'tight' for row in symbolic)
    with_x = sum(row['symbol'] == 'x' for row in symbolic) / sum(row['operation'] == 'mul' for row in symbolic)
    check('symbolic questions vary: half without spaces, most multiplications with x',
          abs(tight - len(symbolic) / 2) <= 2 and 0.5 < with_x < 1, (tight, with_x))


def check_baseline_tool():
    """The pure functions of tools/run_practice_baseline.py (nothing here needs vLLM or the GPU)."""
    section('the baseline tool, without running a model')
    spec = importlib.util.spec_from_file_location('run_practice_baseline',
                                                  os.path.join(cfg.TOOLS_DIR, 'run_practice_baseline.py'))
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    run_path = os.path.join(cfg.practice_run_dir(TINY_MODEL, 'think35', 'baseline_256'), cfg.BASELINE_FILE)
    if os.path.exists(run_path):
        with open(run_path, encoding='utf-8') as f:
            run_ids = [json.loads(line)['id'] for line in f]
        chosen = baseline.choose_problems(cfg.PRACTICE_DIR, 'math', {'gsm8k': 128, 'math': 128}, 0)
        check('the same seed still picks the problems of baseline_256', [p['id'] for p in chosen] == run_ids)
    chosen = baseline.choose_problems(cfg.PRACTICE_DIR, 'basic_arithmetic', {}, 0)
    check('no count means the whole practice pool', len(chosen) == 1000, len(chosen))
    check('grading a short completion', baseline.grade('first_number', ' 4\n', '4', 'length') == (4, True, 'correct')
          and baseline.grade('first_number', ' five', '4', 'length') == (5, False, 'wrong_answer')
          and baseline.grade('first_number', ' the cat', '4', 'length') == (None, False, 'no_number'))
    for key, entry in cfg.TINY_MODEL_REGISTRY.items():
        if os.path.isdir(entry['model_dir']):
            stop_ids = baseline.stop_token_ids(entry['model_dir'], entry)
            check(f'{key}: stop ids are read from its files', bool(stop_ids) and all(isinstance(i, int) for i in stop_ids), stop_ids)
    check('a Windows path becomes a WSL path wherever this file is loaded',
          cfg.to_wsl_path(r'C:\Users\someone\models/Qwen3.5-0.8B') == '/mnt/c/Users/someone/models/Qwen3.5-0.8B'
          and cfg.to_wsl_path('/home/someone/models') == '/home/someone/models')


# ------------------------------------------------------------------------------- stub sessions

def read_session(tier, model):
    session_dir = os.path.join(cfg.SMOKETEST_DIR, f'{tier}_{model}')
    with open(os.path.join(session_dir, cfg.SESSION_STATE_FILE), encoding='utf-8') as f:
        state = json.load(f)
    with open(os.path.join(session_dir, cfg.SESSION_MANIFEST_FILE), encoding='utf-8') as f:
        manifest = json.load(f)
    return session_dir, state, manifest


def tool_call_kinds(session_dir):
    path = os.path.join(session_dir, cfg.SESSION_TOOL_CALLS_FILE)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line)['kind'] for line in f]


def smoke(tier, model, practice_run, **kwargs):
    returned = benchmark.WSP_Benchmark(tier=tier, provider='stub', model=model, practice_run=practice_run,
                                       build_checkpoint=kwargs.pop('build_checkpoint', False), smoketest=True, **kwargs)
    check('a smoketest returns None', returned is None, returned)
    return read_session(tier, model)


def check_stub_sessions(tiny, practice_run):
    first, second = stub.targets(tiny)
    section('stub-spec')
    session_dir, state, manifest = smoke('oneshot_spec', 'stub-spec', practice_run)
    check('status edit_saved', state['status'] == 'edit_saved', state)
    check('expected tensors', state['tensors_edited'] == sorted([first, second]), state['tensors_edited'])
    check('one attempt', state['n_attempts'] == 1, state['n_attempts'])
    check('the versions carry the model label', state['prompt_version'] == f'v1-{TINY_MODEL}'
          and state['dsl_version'] == f'{cfg.DSL_VERSION}-{TINY_MODEL}' and state['task'] == 'math', state)
    first_turn = os.path.join(cfg.SESSION_TURNS_DIR, 't001')
    check('session layout', all(os.path.exists(os.path.join(session_dir, p)) for p in (
        cfg.SESSION_PROMPT_FILE, os.path.join(first_turn, cfg.TURN_ANSWER_FILE),
        os.path.join(first_turn, cfg.TURN_RUN_FILE), cfg.SESSION_MANIFEST_FILE)))
    check('one spec line in tool_calls', tool_call_kinds(session_dir) == ['spec'])
    edited_dir = os.path.join(session_dir, cfg.SESSION_EDITED_DIR)
    check('float32 edits on disk', edits.load_edit(edited_dir, first).dtype.name == 'float32')

    section('stub-spec-invalid-then-valid')
    session_dir, state, manifest = smoke('oneshot_spec', 'stub-spec-invalid-then-valid', practice_run)
    check('status edit_saved', state['status'] == 'edit_saved', state)
    check('two attempts', state['n_attempts'] == 2, state['n_attempts'])
    check('both turns recorded', os.path.exists(os.path.join(session_dir, cfg.SESSION_TURNS_DIR, 't002', cfg.TURN_ANSWER_FILE)))
    check('only the second spec applied', tool_call_kinds(session_dir) == ['spec'])

    section('stub-code')
    session_dir, state, manifest = smoke('oneshot_code', 'stub-code', practice_run)
    check('status edit_saved', state['status'] == 'edit_saved', state)
    check('expected tensor', state['tensors_edited'] == [first], state['tensors_edited'])
    check('16 columns changed', manifest['entries_changed'] == tiny.shapes[first][0] * stub.SCALED_COLUMNS,
          manifest['entries_changed'])
    check('work folder', all(os.path.exists(os.path.join(session_dir, cfg.SESSION_WORK_DIR, f)) for f in (
        kernel_runner.CODE_FILE, kernel_runner.OUTPUT_FILE, kernel_runner.RESULT_FILE)))
    check('one python line in tool_calls', tool_call_kinds(session_dir) == ['python'])
    check('no exception', state['exception'] is None, state['exception'])

    section('stub-code-raises')
    session_dir, state, manifest = smoke('oneshot_code', 'stub-code-raises', practice_run)
    check('the edit before the exception is kept', state['tensors_edited'] == [second], state['tensors_edited'])
    check('status edit_saved', state['status'] == 'edit_saved', state)
    check('the exception is recorded', 'deliberate stub failure' in (state['exception'] or ''), state['exception'])
    check('a raising program is not retried', state['n_attempts'] == 1, state['n_attempts'])

    section('stub-no-block')
    session_dir, state, manifest = smoke('oneshot_spec', 'stub-no-block', practice_run, max_worker_retries=3)
    check('status no_valid_reply', state['status'] == 'no_valid_reply', state)
    check('every attempt used', state['n_attempts'] == 3, state['n_attempts'])
    check('nothing edited', manifest['tensors'] == [], manifest)

    section('a version and a task that do not exist')
    for label, arguments, word in (('an unknown prompt version', {'prompt_version': 'v9'}, 'prompt_version'),
                                   ('a task the model does not have', {'task': 'basic_arithmetic'}, 'task')):
        try:
            benchmark.WSP_Benchmark(provider='stub', model='stub-spec', practice_run=practice_run, smoketest=True, **arguments)
            check(f'{label} raises at entry', False)
        except ValueError as exc:
            check(f'{label} raises at entry', word in str(exc), exc)


def check_other_model_sessions(tiny, runs_dir, run_kernel):
    """One stub session per tier for a model other than Qwen, from its made-up practice run."""
    section(f'{tiny.key}: stub sessions (prompt version {tiny.entry["default_version"]})')
    first, second = stub.targets(tiny)
    real_runs = cfg.PRACTICE_RUNS_DIR
    cfg.PRACTICE_RUNS_DIR = runs_dir
    try:
        session_dir, state, manifest = smoke('oneshot_spec', 'stub-spec', SYNTHETIC_RUN, tiny_model=tiny.key)
        check('oneshot_spec: both tensors edited', state['status'] == 'edit_saved'
              and state['tensors_edited'] == sorted([first, second]), state)
        task, _ = cfg.resolve_task(tiny.key)
        check('the versions carry the model label and the task is recorded',
              state['prompt_version'] == f"{tiny.entry['default_version']}-{tiny.entry['label']}"
              and state['tiny_model'] == tiny.key and state['task'] == task, state)
        with open(os.path.join(session_dir, cfg.SESSION_TOOL_CALLS_FILE), encoding='utf-8') as f:
            directions = json.loads(f.readline())['directions']
        check('the direction records how it was folded',
              directions[0].get('fold_post_norm', False) == bool(tiny.entry['post_norms']), directions)
        if run_kernel:
            _, state, manifest = smoke('oneshot_code', 'stub-code', SYNTHETIC_RUN, tiny_model=tiny.key)
            check('oneshot_code: 16 columns changed through the kernel', state['status'] == 'edit_saved'
                  and manifest['entries_changed'] == tiny.shapes[first][0] * stub.SCALED_COLUMNS, state)
        if tiny.key == SMALLEST_MODEL:
            check_checkpoint(tiny, SYNTHETIC_RUN)
    finally:
        cfg.PRACTICE_RUNS_DIR = real_runs


def check_failures_are_recorded(practice_run):
    section('statuses and failures')
    empty = {'tensors': []}
    check('empty_edit status', session.final_status(empty, session.ACCEPTED) == 'empty_edit')
    check('no_valid_reply status', session.final_status(empty, session.ATTEMPTS_EXHAUSTED) == 'no_valid_reply')
    check('failed status', session.final_status(empty, session.API_FAILED) == 'failed')
    check('saved edits win', session.final_status({'tensors': [1]}, session.ATTEMPTS_EXHAUSTED) == 'edit_saved')

    original_query = stub.query_stub
    async def always_fails(model, prompt, tiny):
        # TimeoutError because ut.report_exc treats it as routine and does not write to error_logs/.
        raise TimeoutError('simulated API failure')
    stub.query_stub = always_fails
    try:
        _, state, _ = smoke('oneshot_spec', 'stub-spec', practice_run, max_worker_retries=2)
    finally:
        stub.query_stub = original_query
    check('API failures end as failed, not as a crash', state['status'] == 'failed', state)
    check('the API error is recorded', 'simulated API failure' in (state['exception'] or ''), state['exception'])

    try:
        benchmark.WSP_Benchmark(provider='stub', model='stub-spec', practice_run=FIXTURE_RUN)
        check('the dev fixture is refused outside a smoketest', False)
    except ValueError as exc:
        check('the dev fixture is refused outside a smoketest', 'smoketest' in str(exc), exc)
    try:
        benchmark.WSP_Benchmark(provider='OpenAI', model='not-a-model', practice_run=practice_run, smoketest=True)
        check('an unregistered model raises at entry', False)
    except ValueError as exc:
        check('an unregistered model raises at entry', 'not registered' in str(exc), exc)


def check_real_route_accepts_our_arguments(tiny, practice_run):
    """The real wrapper's signature against what a session sends it, through the real router. The
    wrapper itself is swapped for a stand-in, so no request leaves the machine."""
    section('the real OpenAI route, with no network')
    entry = registry.BASE_MODEL_REGISTRY['OpenAI']
    real_query, calls = entry['query'], []
    model = list(entry['models'])[0]

    async def stand_in(**kwargs):
        inspect.signature(real_query).bind(**kwargs)  # raises TypeError on an argument it does not take
        calls.append(kwargs)
        return await stub.query_stub('stub-spec', kwargs['prompt'], tiny)

    entry['query'] = stand_in
    try:
        benchmark.WSP_Benchmark(provider='OpenAI', model=model, practice_run=practice_run, build_checkpoint=False,
                                llm_kwargs={'reasoning_effort': 'low', 'host': None, 'stream': True}, smoketest=True)
    finally:
        entry['query'] = real_query
    _, state, _ = read_session('oneshot_spec', benchmark._folder_safe(model))
    check('the wrapper accepts every argument', state['status'] == 'edit_saved', state)
    check('the system prompt is sent', calls and calls[0]['system_prompt'] == prompts.system_prompt('v1'))
    check('llm_kwargs arrive merged', calls and calls[0]['reasoning_effort'] == 'low' and calls[0]['verbose'] is False)


# ------------------------------------------------------------------------------ kernel limits

def check_kernel_limits(tiny):
    section('kernel limits')
    with tempfile.TemporaryDirectory() as session_dir:
        def run(code, **limits):
            work_dir = os.path.join(session_dir, cfg.SESSION_WORK_DIR)
            shutil.rmtree(work_dir, ignore_errors=True)
            return kernel.run(code, work_dir, tiny, limits=limits or None)

        documented = [line.split()[0].split('(')[0] for line in kernel_runner.PRELUDE_DOC.splitlines()]
        program = f'for name in {documented!r}:\n    assert name in globals(), name\nimport torch\nassert not torch.is_grad_enabled()\nprint(len(NAMES))'
        result = run(program)
        check('everything the prompt documents is defined', result['ok'] and result['out'].strip() == str(len(tiny.names)), result)

        result = run('while True:\n    pass', timeout_s=5)
        check('an infinite loop stops at its time limit', not result['ok'] and 'timed out' in (result['exception'] or ''), result)

        result = run('import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nprint("x", flush=True)\nwhile True:\n    pass',
                     timeout_s=3, hard_kill_margin_s=6)
        check('a program ignoring the limit is killed', not result['ok'] and 'exit code' in (result['exception'] or ''), result)
        check('its output survives the kill', result['out'].strip() == 'x', result['out'])

        result = run('import torch\nx = torch.zeros(int(4e9), dtype=torch.float32)', mem_bytes=8 << 30)
        check('the memory limit fires', not result['ok'] and 'memory' in (result['exception'] or '').lower(), result)

        result = run('for i in range(100000):\n    print("0123456789")', max_output_chars=500)
        check('output is cut at the limit', result['out'].endswith('[output cut at 500 characters]') and len(result['out']) < 600, len(result['out']))

        try:
            run('pass', mode='bwrap')
            check('the bwrap switch is visible and refuses', False)
        except NotImplementedError as exc:
            check('the bwrap switch is visible and refuses', str(exc) == 'v2', exc)


# -------------------------------------------------------------------------- the real write path

def write_synthetic_practice_run(runs_dir, tiny_model, n_failing=12):
    """A small run made of genuine practice problems of the model's default task, so sessions can be
    tested without a GPU baseline. Its answers are made up; it exists only in a temporary folder."""
    task, protocol = cfg.resolve_task(tiny_model)
    grader = cfg.TASKS[task]['grader']
    with open(os.path.join(cfg.PRACTICE_DIR, cfg.practice_files(task)[-1]), encoding='utf-8') as f:
        problems = [json.loads(next(f)) for _ in range(n_failing)]
    run_dir = cfg.practice_run_dir(tiny_model, protocol, SYNTHETIC_RUN)
    run_dir = os.path.join(runs_dir, os.path.relpath(run_dir, cfg.PRACTICE_RUNS_DIR))
    os.makedirs(run_dir)
    text, n_tokens = ('made-up answer ' * 100, 400) if grader == 'boxed' else (' 1000 made up', 4)
    rows = [{**{key: p.get(key) for key in ('id', 'source', 'problem', 'answer', 'solution')},
             'text': text, 'n_tokens': n_tokens, 'finish_reason': 'stop', 'extracted': '1' if grader == 'boxed' else 1000,
             'correct': False, 'outcome': 'wrong_answer'} for p in problems]
    with open(os.path.join(run_dir, cfg.BASELINE_FILE), 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(row) + '\n' for row in rows)
    counts = {'n': len(rows), 'n_correct': 0, 'accuracy': 0.0, 'mean_tokens': float(n_tokens), 'cut_share': 0.0}
    failures = {'boxed': {'wrong_answer': len(rows), 'no_answer_finished': 0, 'cut_at_max_tokens': 0},
                'first_number': {'wrong_answer': len(rows), 'no_number': 0}}[grader]
    summary = {**counts, 'by_source': {rows[0]['source']: counts}, 'failures': failures,
               'sampling': cfg.PROTOCOLS[protocol]['sampling']}
    with open(os.path.join(run_dir, cfg.SUMMARY_FILE), 'w', encoding='utf-8') as f:
        json.dump(summary, f)
    return SYNTHETIC_RUN


async def _two_concurrent_sessions(practice_run):
    return await asyncio.gather(*[
        benchmark.WSP_Benchmark_async(tier=tier, provider='stub', model=model, practice_run=practice_run,
                                      build_checkpoint=False, Notes='selftest')
        for tier, model in (('oneshot_spec', 'stub-spec'), ('oneshot_code', 'stub-code'))])


def check_master_rows(runs_dir):
    section('master rows and IDs (in a temporary folder)')
    real = {name: getattr(cfg, name) for name in ('MASTER_CSV', 'SESSIONS_DIR', 'PRACTICE_RUNS_DIR')}
    with tempfile.TemporaryDirectory() as temp:
        cfg.MASTER_CSV = os.path.join(temp, 'results', 'Master_WSP_Edits.csv')
        cfg.SESSIONS_DIR = os.path.join(temp, 'results', 'sessions')
        cfg.PRACTICE_RUNS_DIR = runs_dir
        try:
            ids = asyncio.run(_two_concurrent_sessions(SYNTHETIC_RUN))
            check('concurrent sessions get distinct IDs', sorted(ids) == [0, 1], ids)
            next_id = benchmark.WSP_Benchmark(provider='stub', model='stub-no-block', practice_run=SYNTHETIC_RUN, build_checkpoint=False)
            check('the next ID follows', next_id == 2, next_id)

            rows = pl.read_csv(cfg.MASTER_CSV, schema=benchmark.MASTER_SCHEMA).sort('ID')
            check('one row per run', rows.height == 3, rows.height)
            check('columns match the schema', rows.columns == list(benchmark.MASTER_SCHEMA), rows.columns)
            check('statuses', rows['status'].to_list() == ['edit_saved', 'edit_saved', 'no_valid_reply'], rows['status'].to_list())
            check('versions and task are recorded', rows['prompt_version'][0] == f'v1-{TINY_MODEL}'
                  and rows['dsl_version'][0] == f'{cfg.DSL_VERSION}-{TINY_MODEL}' and rows['task'][0] == 'math',
                  rows.select('prompt_version', 'dsl_version', 'task').row(0))
            misc = json.loads(rows['misc'][0])
            check('misc has the example ids, the protocol and Notes', len(misc['example_ids']) == cfg.DIAG_DEFAULTS['k']
                  and misc['Notes'] == 'selftest' and misc['protocol'] == 'think35', misc)
            folders = [os.path.basename(path) for path in rows['session_dir'].to_list()]
            check('each row names its own session folder',
                  sorted(folders) == sorted(os.listdir(cfg.SESSIONS_DIR)) and all(
                      name.startswith(f'{row_id}_') for row_id, name in zip(rows['ID'].to_list(), folders)), folders)

            # A master CSV from before the task column existed must still be readable.
            rows.drop('task').write_csv(cfg.MASTER_CSV)
            reread = benchmark._read_master()
            check('a master CSV without the task column reads as the math task',
                  reread.columns == list(benchmark.MASTER_SCHEMA) and reread['task'].to_list() == ['math'] * 3,
                  reread.columns)
        finally:
            for name, value in real.items():
                setattr(cfg, name, value)


# ---------------------------------------------------------------------------------- checkpoints

def check_checkpoint(tiny, practice_run):
    size_gb = os.path.getsize(tiny.weights_path) / 2**30
    section(f'{tiny.key}: built checkpoint ({size_gb:.2f} GB)')
    first, second = stub.targets(tiny)
    smoke('oneshot_spec', 'stub-spec', practice_run, build_checkpoint=True, tiny_model=tiny.key)
    out_dir = os.path.join(cfg.SMOKETEST_CHECKPOINT_DIR, 'stub-spec')
    path = os.path.join(out_dir, cfg.CHECKPOINT_WEIGHTS_FILE)
    check('the checkpoint file exists', os.path.exists(path), out_dir)
    weights = [name for name in os.listdir(out_dir) if name.endswith(cfg.WEIGHTS_EXTENSIONS)]
    check('it is the only weights file, with no index pointing at an old one',
          weights == [cfg.CHECKPOINT_WEIGHTS_FILE] and not os.path.exists(os.path.join(out_dir, cfg.CHECKPOINT_NOT_COPIED[0])), weights)
    record = cfg.read_checkpoint_record(out_dir)
    task, _ = cfg.resolve_task(tiny.key)
    check('edits.json names the tiny model and the task',
          record is not None and record['tiny_model'] == tiny.key and record['task'] == task, record)
    check('the config and the tokenizer came along', all(os.path.exists(os.path.join(out_dir, name)) for name in (
        cfg.MODEL_CONFIG_FILE, cfg.MODEL_TOKENIZER_FILE)))

    with safe_open(tiny.weights_path, framework='pt') as base, safe_open(path, framework='pt') as edited:
        check('metadata format pt', edited.metadata() == {'format': 'pt'}, edited.metadata())
        check('the same tensor names', set(base.keys()) == set(edited.keys()))
        differing, same_dtype = [], True
        for name in base.keys():
            a, b = base.get_tensor(name), edited.get_tensor(name)
            same_dtype = same_dtype and a.dtype == b.dtype and a.shape == b.shape
            if not torch.equal(a, b):
                differing.append(name)
    check('every dtype and shape is unchanged', same_dtype)
    check('only the edited tensors differ', sorted(differing) == sorted([first, second]), differing)
    benchmark._clear_folder(cfg.SMOKETEST_CHECKPOINT_DIR)


# ----------------------------------------------------------------------------------------- main

def default_practice_run():
    fixture = cfg.practice_run_dir(TINY_MODEL, 'think35', FIXTURE_RUN)
    return FIXTURE_RUN if os.path.isdir(fixture) else 'baseline_256'


def check_qwen(args, runs_dir):
    tiny = tiny_model_module.open(TINY_MODEL)
    check_edit_rules(tiny)
    check_dsl(tiny)
    check_prompts(tiny, runs_dir)
    check_reference_prompts(tiny)
    check_stub_sessions(tiny, args.practice_run)
    check_failures_are_recorded(args.practice_run)
    check_real_route_accepts_our_arguments(tiny, args.practice_run)
    if not args.skip_kernel_limits:
        check_kernel_limits(tiny)
    check_master_rows(runs_dir)
    if args.build_checkpoint:
        check_checkpoint(tiny, args.practice_run)


def check_other_model(key, args, runs_dir):
    tiny = tiny_model_module.open(key)
    check_edit_rules(tiny)
    check_dsl(tiny)
    check_prompts(tiny, runs_dir)
    check_reference_prompts(tiny)
    check_other_model_sessions(tiny, runs_dir, run_kernel=not args.skip_kernel_limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--practice-run', default=default_practice_run())
    parser.add_argument('--build-checkpoint', action='store_true')
    parser.add_argument('--skip-kernel-limits', action='store_true',
                        help='skip the checks that start the oneshot_code kernel, except the Qwen stub-code sessions')
    parser.add_argument('--only-model', default=None, help='check one tiny model of the registry only')
    args = parser.parse_args()

    models = [key for key in cfg.TINY_MODEL_REGISTRY if args.only_model in (None, key)]
    real_master_before = os.path.getmtime(cfg.MASTER_CSV) if os.path.exists(cfg.MASTER_CSV) else None
    with tempfile.TemporaryDirectory() as runs_dir:
        try:
            check_block_parsing()
            check_templates()
            check_arithmetic()
            check_baseline_tool()
            for key in models:
                if not os.path.isdir(cfg.TINY_MODEL_REGISTRY[key]['model_dir']):
                    section(f"{key}: NOT CHECKED, its folder {cfg.TINY_MODEL_REGISTRY[key]['model_dir']} does not "
                            f'exist (docs/SETUP.md says how to get it)')
                    _not_checked.append(key)
                    continue
                write_synthetic_practice_run(runs_dir, key)
                if key == TINY_MODEL:
                    check_qwen(args, runs_dir)
                else:
                    check_other_model(key, args, runs_dir)
        except Exception:
            traceback.print_exc()
            _failures.append('selftest crashed')

    real_master_after = os.path.getmtime(cfg.MASTER_CSV) if os.path.exists(cfg.MASTER_CSV) else None
    section('the real master CSV')
    check('not touched by the selftest', real_master_before == real_master_after)
    # The stub edits are about 30 MB, so they do not stay behind.
    benchmark._clear_folder(cfg.SMOKETEST_DIR)

    if _not_checked:
        print(f'\nNot checked, because their model folders are missing: {_not_checked}')
    print(f'\n{len(_failures)} check(s) failed: {_failures}' if _failures else '\nAll checks passed.')
    sys.exit(1 if _failures else 0)


if __name__ == '__main__':
    main()
