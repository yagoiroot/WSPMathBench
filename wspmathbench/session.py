"""One session of one tier with one primary LLM: query, parse, apply or run, record.

Where it sits in the flow: called by benchmark.py, which reserves the session's ID and folder
and appends the master row. The steps of a session, marked below and referred to by number in the
other modules:
    1  open the tiny model (tiny_model.py)
    2  build the diagnostic pack from the practice run (diagnostics.py)
    3  render the prompt (prompts.py)
    4  query the primary LLM, parse its one block, apply the edit specification (dsl.py) or run
       the program (kernel.py); a reply that cannot be used goes back with the reasons
    5  write the edit manifest (edits.py)
    6  set the status
    7  build the edited checkpoint (checkpoint.py)
    8  append the master row (benchmark.py)

Writes the session folder: prompt.txt, turns/t001/{answer.md, run.json, reasoning.md},
tool_calls.jsonl, work/, edited/, edit/manifest.json, state.json. Writes the full checkpoint to
checkpoint_dir (under cfg.CHECKPOINT_DIR).

A failure inside a session is an outcome: it is recorded in the returned record and in state.json
with whatever cost was already spent. Misconfiguration raises earlier, in `prepare` or in
benchmark, before any money is spent or any ID is used.
"""
from wspmathbench import config as cfg

import asyncio
import json
import os
import re
import time

from wspmathbench.llm import registry
from wspmathbench.llm import utils as ut

from wspmathbench import checkpoint
from wspmathbench import diagnostics
from wspmathbench import dsl
from wspmathbench import edits
from wspmathbench import kernel
from wspmathbench import prompts
from wspmathbench import stub
from wspmathbench import tiny_model as tiny_model_module

STUB_PROVIDER = 'stub'
_FENCE = re.compile(r'^\s{0,3}```(\S*)\s*$')

# How the attempts loop ended.
ACCEPTED, ATTEMPTS_EXHAUSTED, API_FAILED = 'accepted', 'attempts_exhausted', 'api_failed'


def prepare(tiny_model, practice_run, diag, task=None):
    """Steps 1-2: (tiny, pack). Separate from run_session so a missing model or practice run
    raises before an ID is used. task=None means the tiny model's default task."""
    tiny = tiny_model_module.open(tiny_model)
    pack = diagnostics.build(tiny_model, practice_run, diag, task)
    return tiny, pack


def parse_single_block(text, expected_kind):
    """(the content of the one fenced block of that kind, None), or (None, a sentence saying why not).

    Fences of other kinds in the reasoning do not count. A fence that is never closed is no block.
    """
    blocks, open_kind, lines = [], None, []
    for line in text.splitlines():
        fence = _FENCE.match(line)
        if open_kind is None:
            if fence:
                open_kind, lines = fence.group(1).lower(), []
        elif fence and not fence.group(1):
            if open_kind == expected_kind:
                blocks.append('\n'.join(lines))
            open_kind = None
        else:
            lines.append(line)
    if len(blocks) == 1:
        return blocks[0], None
    if not blocks:
        return None, f'the reply has no complete ```{expected_kind} block'
    return None, f'the reply has {len(blocks)} ```{expected_kind} blocks, exactly one is required'


def final_status(manifest, attempts_outcome):
    """Step 6. Edits that were saved count even if the program raised afterwards."""
    if manifest['tensors']:
        return 'edit_saved'
    if attempts_outcome == API_FAILED:
        return 'failed'
    return 'no_valid_reply' if attempts_outcome == ATTEMPTS_EXHAUSTED else 'empty_edit'


async def run_session(settings, tiny, pack, session_dir, checkpoint_dir, session_id):
    """Steps 3-7. Returns the record that benchmark turns into the master row.

    settings: {'tier', 'task', 'prompt_version', 'provider', 'model', 'llm_kwargs',
    'max_worker_retries', 'retry_wait', 'build_checkpoint'}. task and prompt_version are resolved
    names (cfg.resolve_task, cfg.resolve_version), never None.
    record: {'status', 'n_attempts', 'manifest', 'total_cost', 'token_list', 'wall_s',
    'checkpoint_dir', 'kernel_seconds', 'exception', 'server_providers'}.
    """
    started = time.time()
    record = {'status': 'failed', 'n_attempts': 0, 'manifest': {'tensors': [], 'n_tensors': 0, 'entries_changed': 0},
              'total_cost': 0.0, 'token_list': [], 'wall_s': None, 'checkpoint_dir': '',
              'kernel_seconds': None, 'exception': None, 'server_providers': []}
    try:
        await _run_steps(settings, tiny, pack, session_dir, checkpoint_dir, session_id, record)
    except Exception as exc:
        record['status'] = 'failed'
        record['exception'] = ut.report_exc(exc, f'WSP session {session_dir}')
        print(f"WSP session failed: {record['exception']}")
    record['wall_s'] = round(time.time() - started, 2)
    _write_state(session_dir, settings, tiny, record, session_id)
    return record


async def _run_steps(settings, tiny, pack, session_dir, checkpoint_dir, session_id, record):
    edited_dir = os.path.join(session_dir, cfg.SESSION_EDITED_DIR)
    os.makedirs(edited_dir, exist_ok=True)

    # Step 3: the prompt.
    paths = {'model_dir': cfg.to_wsl_path(tiny.model_dir), 'practice_dir': cfg.to_wsl_path(cfg.PRACTICE_DIR),
             'work_dir': cfg.to_wsl_path(os.path.join(session_dir, cfg.SESSION_WORK_DIR))}
    prompt = prompts.render(settings['tier'], tiny, pack, paths, settings['prompt_version'])
    _write_text(os.path.join(session_dir, cfg.SESSION_PROMPT_FILE), prompt)

    # Step 4: query, parse, apply or run.
    outcome = await _run_attempts(settings, tiny, prompt, session_dir, record)

    # Steps 5-6: the manifest and the status.
    manifest = await asyncio.to_thread(edits.manifest, edited_dir, tiny)
    record['manifest'] = manifest
    record['status'] = final_status(manifest, outcome)
    _write_json(os.path.join(session_dir, cfg.SESSION_MANIFEST_FILE), manifest)
    print(f"status: {record['status']}, tensors edited: {manifest['n_tensors']}, entries changed: {manifest['entries_changed']:,}")

    # Step 7: the checkpoint. A failure here keeps the status: the edit itself is safely saved.
    if settings['build_checkpoint'] and record['status'] == 'edit_saved':
        await _build_checkpoint(tiny, edited_dir, checkpoint_dir, manifest, session_id, settings['task'], record)


async def _run_attempts(settings, tiny, prompt, session_dir, record):
    """The loop of step 4. Returns ACCEPTED, ATTEMPTS_EXHAUSTED or API_FAILED."""
    expected_kind = cfg.EXPECTED_BLOCK_KIND[settings['tier']]
    prompt_this_attempt = prompt
    for attempt in range(1, settings['max_worker_retries'] + 1):
        print(f"[{settings['tier']} {settings['model']}] attempt {attempt}: querying")
        full = await _query_with_retries(settings, prompt_this_attempt, tiny, attempt, session_dir, record)
        if full is None:
            return API_FAILED
        block, problem = parse_single_block(full['text'], expected_kind)
        errors = [problem] if problem else await _use_block(settings, block, tiny, session_dir, record)
        if not errors:
            return ACCEPTED
        print(f'attempt {attempt}: {len(errors)} problem(s), first: {errors[0]}')
        prompt_this_attempt = prompts.retry_prompt(prompt, full['text'], errors)
    return ATTEMPTS_EXHAUSTED


async def _query_with_retries(settings, prompt, tiny, attempt, session_dir, record):
    """One reply through the shared retry ladder (API failures only), recorded. None if it failed."""
    started = time.time()

    def call():
        # Live footer (ut.live_status): every call, a retry after an API failure included, restarts
        # the session's timer and count; the footer shows the attempt number from the second on.
        ut.live_status_attempt(attempt - 1)
        return _query(settings, prompt, tiny)

    full, lost_cost, last_error = await registry.retry_llm_call(
        call, max_retries=settings['max_worker_retries'],
        label='WSP session', retry_wait=settings['retry_wait'])
    record['total_cost'] += float(lost_cost)
    if full is None:
        record['exception'] = last_error
        return None
    record['n_attempts'] = attempt
    record['total_cost'] += float(full.get('cost_estimate') or 0.0)
    record['token_list'].append(_token_tuple(full))
    record['server_providers'].append(full.get('server_provider'))
    _write_turn(session_dir, attempt, settings, full, lost_cost, round(time.time() - started, 2))
    return full


def _query(settings, prompt, tiny):
    """A fresh coroutine for one call: the stub, or the shared routing layer."""
    if settings['provider'] == STUB_PROVIDER:
        return stub.query_stub(settings['model'], prompt, tiny)
    return registry.query_base_model(settings['provider'], settings['model'], prompt,
                                     system_prompt=prompts.system_prompt(settings['prompt_version']),
                                     **settings['llm_kwargs'])


async def _use_block(settings, block, tiny, session_dir, record):
    """Apply a spec or run a program. Returns the errors that send the reply back (a program that
    runs and raises sends nothing back: that is a legitimate outcome, not a retry)."""
    if settings['tier'] == 'oneshot_spec':
        return await _apply_spec(block, tiny, session_dir)
    await _run_program(block, tiny, session_dir, record)
    return []


async def _apply_spec(block, tiny, session_dir):
    spec, errors = dsl.parse(block)
    if not errors:
        errors = await asyncio.to_thread(dsl.validate, spec, tiny)
    if errors:
        return errors
    edited_dir = os.path.join(session_dir, cfg.SESSION_EDITED_DIR)
    tool_call, errors = await asyncio.to_thread(dsl.apply, spec, tiny, edited_dir)
    if not errors:
        _append_tool_call(session_dir, tool_call)
    return errors


async def _run_program(block, tiny, session_dir, record):
    work_dir = os.path.join(session_dir, cfg.SESSION_WORK_DIR)
    result = await asyncio.to_thread(kernel.run, block, work_dir, tiny)
    record['kernel_seconds'] = result['seconds']
    record['exception'] = result['exception']
    print(f"kernel: ok={result['ok']}, {result['seconds']} s, edited: {len(result['edited'])}, exception: {result['exception']}")
    _append_tool_call(session_dir, {'kind': 'python', 'code': block, 'out': result['out'], 's': result['seconds'],
                                    'ok': result['ok'], 'exception': result['exception'], 'edited': result['edited']})


async def _build_checkpoint(tiny, edited_dir, checkpoint_dir, manifest, session_id, task, record):
    try:
        print(f'building checkpoint: {checkpoint_dir}')
        await asyncio.to_thread(checkpoint.build, tiny, edited_dir, checkpoint_dir, manifest, session_id, task)
        record['checkpoint_dir'] = checkpoint_dir
    except Exception as exc:
        record['exception'] = ut.report_exc(exc, f'WSP checkpoint {checkpoint_dir}')
        print(f"checkpoint failed: {record['exception']}")


def _token_tuple(full):
    return tuple(full.get(key) or 0 for key in
                 ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens'))


def _write_turn(session_dir, attempt, settings, full, lost_cost, wall_s):
    turn_dir = os.path.join(session_dir, cfg.SESSION_TURNS_DIR, f't{attempt:03d}')
    _write_text(os.path.join(turn_dir, cfg.TURN_ANSWER_FILE), full['text'])
    if full.get('reasoning_text'):
        _write_text(os.path.join(turn_dir, cfg.TURN_REASONING_FILE), full['reasoning_text'])
    _write_json(os.path.join(turn_dir, cfg.TURN_RUN_FILE), {
        'provider': settings['provider'], 'model': settings['model'], 'llm_kwargs': settings['llm_kwargs'],
        'tokens': dict(zip(('prompt', 'cached', 'completion', 'reasoning', 'total'), _token_tuple(full))),
        'cost': full.get('cost_estimate'), 'lost_cost': lost_cost, 'wall_s': wall_s,
        'server_provider': full.get('server_provider'), 'status': full.get('status'),
        'incomplete_reason': full.get('incomplete_reason')})


def version_labels(prompt_version, tiny_model):
    """What a session records as its versions (the master row and state.json say the same)."""
    return {'benchmark_version': cfg.BENCHMARK_VERSION,
            'prompt_version': cfg.version_label(prompt_version, tiny_model),
            'dsl_version': cfg.version_label(cfg.DSL_VERSION, tiny_model)}


def _write_state(session_dir, settings, tiny, record, session_id):
    manifest = record['manifest']
    _write_json(os.path.join(session_dir, cfg.SESSION_STATE_FILE), {
        'session_id': session_id, 'tier': settings['tier'], 'tiny_model': tiny.key, 'task': settings['task'],
        'provider': settings['provider'], 'model': settings['model'],
        'status': record['status'], 'n_attempts': record['n_attempts'], 'total_cost': record['total_cost'],
        'wall_s': record['wall_s'], 'kernel_seconds': record['kernel_seconds'], 'exception': record['exception'],
        'tensors_edited': [t['name'] for t in manifest['tensors']], 'checkpoint_dir': record['checkpoint_dir'],
        **version_labels(settings['prompt_version'], tiny.key)})


def _append_tool_call(session_dir, tool_call):
    with open(os.path.join(session_dir, cfg.SESSION_TOOL_CALLS_FILE), 'a', encoding='utf-8') as f:
        f.write(json.dumps(tool_call, cls=ut.NumpyEncoder) + '\n')


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _write_json(path, data):
    _write_text(path, json.dumps(data, cls=ut.NumpyEncoder, indent=1))
