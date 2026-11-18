"""WSP_Benchmark / WSP_Benchmark_async: the entry point. One call is one edit session.

Where it sits in the flow: checks the arguments, runs steps 1-2 of a session through
session.prepare, reserves the session's ID and folder, hands over to session.run_session
(steps 3-7), then appends the master row (step 8); session.py lists the steps. Scores are not
computed here: llm/scoring.py scores the checkpoint a session built, and its rows join on the
session's ID.

Reads and writes <results>/Master_WSP_Edits.csv (cfg.MASTER_CSV). Creates
<results>/sessions/<ID>_<tier>_<tiny model>_<model>_<reasoning effort>/, or clears and reuses
<results>/smoketest/ for a smoketest.
"""
from wspmathbench import config as cfg

import asyncio
import datetime
import json
import os
import re
import shutil
import time

import polars as pl

from wspmathbench.llm import registry
from wspmathbench.llm import utils as ut

from wspmathbench import session
from wspmathbench import stub

# Merged under the caller's llm_kwargs; the merged dict is recorded verbatim, because it is the
# run's treatment. Changing a default changes the recorded metadata of every future row.
WSP_LLM_DEFAULTS = {
    'OpenAI': {'reasoning_effort': None, 'max_tokens': None, 'temperature': None, 'top_p': None,
               'stream': False, 'verbose': False},
    'Anthropic': {'verbose': False},
    'Google': {'verbose': False},
    'DeepSeek': {'verbose': False},
    'Kimi': {'verbose': False},
    session.STUB_PROVIDER: {},
}

MASTER_SCHEMA = {
    'ID': pl.Int64, 'date_time': pl.String,
    'benchmark_version': pl.String, 'prompt_version': pl.String, 'dsl_version': pl.String,
    'tier': pl.String, 'tiny_model': pl.String, 'tiny_revision': pl.String, 'task': pl.String,
    'provider': pl.String, 'model': pl.String, 'llm_kwargs': pl.String,
    'status': pl.String, 'n_attempts': pl.Int64, 'tensors_edited': pl.String,
    'entries_changed': pl.Int64, 'relative_change': pl.String, 'total_cost': pl.Float64,
    'token_list': pl.String, 'wall_s': pl.Float64, 'session_dir': pl.String,
    'checkpoint_dir': pl.String, 'misc': pl.String,
}

# Columns added after rows existed, with the value every earlier row had. A master CSV written
# before a column existed is read with that value filled in, and gains the column on the next append.
COLUMNS_ADDED_LATER = {'task': 'math'}

_CLEAR_ATTEMPTS = 8
_CLEAR_BASE_DELAY_S = 0.25

# The practice run each tiny model's sessions read unless told otherwise: the unedited model's
# answers to 256 practice problems, or to the whole pool of 1,000 for BasicArithmetic.
MODEL_PRACTICE_RUNS = {
    'Qwen3.5-0.8B': 'baseline_256', 'gemma-3-270m-it': 'baseline_256', 'LFM2.5-350M': 'baseline_256',
    'MobileLLM-R1-950M': 'baseline_256', 'TinyStories-33M': 'baseline_1000',
}


def WSP_Benchmark(tier='oneshot_spec', tiny_model='Qwen3.5-0.8B',
                  provider='OpenAI', model='gpt-5-nano', llm_kwargs=None,
                  diag=None, max_worker_retries=3, retry_wait='strat2',
                  build_checkpoint=True, Notes=None, smoketest=False, task=None, prompt_version=None,
                  live_status=None, practice_run=None):
    """Synchronous WSP_Benchmark_async (same arguments, same return value)."""
    return asyncio.run(WSP_Benchmark_async(
        tier=tier, tiny_model=tiny_model, task=task, prompt_version=prompt_version,
        provider=provider, model=model,
        llm_kwargs=llm_kwargs, diag=diag, max_worker_retries=max_worker_retries, retry_wait=retry_wait,
        build_checkpoint=build_checkpoint, Notes=Notes, smoketest=smoketest, live_status=live_status,
        practice_run=practice_run))


async def WSP_Benchmark_async(tier='oneshot_spec', tiny_model='Qwen3.5-0.8B',
                              provider='OpenAI', model='gpt-5-nano', llm_kwargs=None,
                              diag=None, max_worker_retries=3, retry_wait='strat2',
                              build_checkpoint=True, Notes=None, smoketest=False, task=None,
                              prompt_version=None, live_status=None, practice_run=None):
    """One session of one tier with one primary model. Returns the new row's ID (None for a
    smoketest). Scores are not computed here.

    Parameters
    ----------
    tier : 'oneshot_spec' or 'oneshot_code'
    tiny_model : a key of cfg.TINY_MODEL_REGISTRY
    task : a task of that tiny model (cfg.TASKS); None means its default task (math word problems,
        or basic arithmetic for TinyStories-33M). The task decides the protocol, the practice pool
        and the task sentences of the prompt.
    prompt_version : the scope of the prompt, a member of cfg.PROMPT_VERSIONS ('v1' is written
        around repairing a fault, 'v2' is open-ended); None means the tiny model's default (v1 for
        Qwen3.5-0.8B, v2 for the others). The row records it with the model's label appended.
    practice_run : the folder under data/practice_runs/<tiny_model>/<protocol>/ with the unedited
        model's practice answers; None means the model's entry in MODEL_PRACTICE_RUNS, which is what
        the paper's sessions used. A name starting with DEVFIXTURE is refused unless smoketest=True.
    provider, model : routed through registry.query_base_model. provider='stub' with a model of
        stub.STUB_MODELS returns canned replies and costs nothing.
    llm_kwargs : merged over WSP_LLM_DEFAULTS[provider]; the merged dict is recorded verbatim.
    diag : merged over cfg.DIAG_DEFAULTS (k, head_chars, tail_chars, seed); recorded in misc.
    max_worker_retries : attempts for unparseable replies, and for each LLM call's API failures.
    retry_wait : 'strat1' or 'strat2', as in registry.retry_llm_call.
    build_checkpoint : build the full checkpoint directory after the session (up to about 2 GB,
        under cfg.CHECKPOINT_DIR). Scoring needs it.
    Notes : free text into misc['Notes'].
    smoketest : run everything into <results>/smoketest/ (cleared first) and write no master row.
        Clearing makes concurrent smoketests unsafe, so run them one at a time.
    live_status : None, False, True, 'tokens' or 'chars'
        Live status footer (ut.live_status); None -> ut.LIVE_STATUS_DEFAULT. Display only, not saved.
        The session is the footer's one entry, shown by its ID (0 for a smoketest): elapsed time
        until the reply starts streaming, then the reply's running size (needs stream=True in
        llm_kwargs). The count restarts with each new call and stays put while the edit is applied
        or the program runs and the checkpoint is built. Concurrent sessions each add a line.
    """
    cfg.tiny_entry(tiny_model)  # raises for a name that is not in the registry
    practice_run = MODEL_PRACTICE_RUNS[tiny_model] if practice_run is None else practice_run
    date_time = str(datetime.datetime.now())
    _check_arguments(tier, practice_run, provider, model, llm_kwargs, smoketest)
    task, _ = cfg.resolve_task(tiny_model, task)
    prompt_version = cfg.resolve_version(tiny_model, prompt_version)
    llm_kwargs = {**WSP_LLM_DEFAULTS[provider], **(llm_kwargs or {})}
    registry.check_thinking_arm(llm_kwargs.get('base_model', model), llm_kwargs.get('thinking'))
    diag = {**cfg.DIAG_DEFAULTS, **(diag or {})}
    print(f'llm_kwargs: {llm_kwargs}')

    tiny, pack = await asyncio.to_thread(session.prepare, tiny_model, practice_run, diag, task)
    if smoketest:
        session_id, session_dir, checkpoint_dir = None, *await asyncio.to_thread(_fresh_smoketest_folders, tier, model)
    else:
        session_id, session_dir, checkpoint_dir = await _reserve_session(tier, tiny_model, model, llm_kwargs)
    print(f"session {session_id}: {session_dir} ({tiny_model}, task {task}, protocol {pack['protocol']}, "
          f"prompt {cfg.version_label(prompt_version, tiny_model)})")

    settings = {'tier': tier, 'task': task, 'prompt_version': prompt_version,
                'provider': provider, 'model': model, 'llm_kwargs': llm_kwargs,
                'max_worker_retries': max_worker_retries, 'retry_wait': retry_wait,
                'build_checkpoint': build_checkpoint}
    # Live footer (ut.live_status). The session runs as its own task so that the footer index stays
    # private to it; session.py marks the attempts.
    with ut.live_status(f'WSP {tier} {model}', 1, live_status) as status:
        record = await asyncio.create_task(ut.live_status_task(
            status, session_id or 0,
            session.run_session(settings, tiny, pack, session_dir, checkpoint_dir, session_id)))
    print(f"status: {record['status']}, total_cost: {record['total_cost']}, wall_s: {record['wall_s']}")

    if smoketest:
        return None
    row = _master_row(session_id, date_time, settings, tiny, session_dir, record,
                      _misc(Notes, diag, practice_run, pack, record))
    async with ut._save_lock:
        _append_master_row(row)
    print(f'Saved row {session_id}')
    return session_id


def _check_arguments(tier, practice_run, provider, model, llm_kwargs, smoketest):
    """Misconfiguration raises here, once: inside the retry ladder it would only be retried."""
    if tier not in cfg.TIERS:
        raise ValueError(f'`tier` must be one of {cfg.TIERS}, got {tier!r}.')
    if practice_run.startswith(cfg.DEV_FIXTURE_PREFIX) and not smoketest:
        raise ValueError(f"Practice run '{practice_run}' holds evaluation-set problems. It may only be "
                         f"used with smoketest=True.")
    if provider not in WSP_LLM_DEFAULTS:
        raise ValueError(f'`provider` must be one of {list(WSP_LLM_DEFAULTS)}, got {provider!r}.')
    if provider == session.STUB_PROVIDER:
        if model not in stub.STUB_MODELS:
            raise ValueError(f'With the stub provider, `model` must be one of {stub.STUB_MODELS}, got {model!r}.')
        return
    registered = registry.BASE_MODEL_REGISTRY[provider]['models']
    if model not in registered:
        raise ValueError(f"Model {model!r} is not registered for provider '{provider}'. Registered: {registered}")


def _folder_safe(name):
    return re.sub(r'[^A-Za-z0-9._-]', '-', name)


# ------------------------------------------------------------------------------ session folders

def session_folder_name(session_id, tier, tiny_model, model, llm_kwargs):
    """<ID>_<tier>_<tiny model>_<primary model>_<reasoning effort>. The effort is the
    `reasoning_effort` of the merged llm_kwargs, or 'default' when none was set (the provider's own
    default then applies). The ID stays first: _next_id reads it from there."""
    effort = (llm_kwargs or {}).get('reasoning_effort') or 'default'
    return '_'.join([str(session_id), tier, _folder_safe(tiny_model), _folder_safe(model), _folder_safe(str(effort))])


async def _reserve_session(tier, tiny_model, model, llm_kwargs):
    """(ID, session folder, checkpoint folder). The ID comes first because the session folder and
    the checkpoint are named by it, and concurrent sessions finish in any order. So it is one more
    than the largest ID among the master rows AND the existing folders, and mkdir claims it."""
    async with ut._save_lock:
        session_id = _next_id()
        session_dir = os.path.join(cfg.SESSIONS_DIR, session_folder_name(session_id, tier, tiny_model, model, llm_kwargs))
        os.makedirs(session_dir)
    return session_id, session_dir, os.path.join(cfg.CHECKPOINT_DIR, str(session_id))


def _next_id():
    used = _read_master()['ID'].to_list()
    if os.path.isdir(cfg.SESSIONS_DIR):
        used += [int(name.split('_')[0]) for name in os.listdir(cfg.SESSIONS_DIR) if name.split('_')[0].isdigit()]
    return max(used) + 1 if used else 0


def _fresh_smoketest_folders(tier, model):
    """(session folder, checkpoint folder), after clearing everything from the last smoketest."""
    for folder in (cfg.SMOKETEST_DIR, cfg.SMOKETEST_CHECKPOINT_DIR):
        _clear_folder(folder)
    session_dir = os.path.join(cfg.SMOKETEST_DIR, f'{tier}_{_folder_safe(model)}')
    os.makedirs(session_dir, exist_ok=True)
    return session_dir, os.path.join(cfg.SMOKETEST_CHECKPOINT_DIR, _folder_safe(model))


def _clear_folder(folder):
    # A file-sync client or a virus scanner can hold a new file open for a moment, which fails
    # rmtree on Windows, so a refusal is retried a few times.
    for attempt in range(_CLEAR_ATTEMPTS):
        try:
            shutil.rmtree(folder)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == _CLEAR_ATTEMPTS - 1:
                raise
            time.sleep(_CLEAR_BASE_DELAY_S * 2 ** attempt)


# ---------------------------------------------------------------------------------- master row

def _misc(notes, diag, practice_run, pack, record):
    return {'Notes': notes, 'diag': diag, 'practice_run': practice_run, 'protocol': pack['protocol'],
            'example_ids': pack['example_ids'],
            'sandbox_mode': cfg.SANDBOX['mode'], 'kernel_seconds': record['kernel_seconds'],
            'exception': record['exception'], 'server_provider': record['server_providers']}


def _master_row(session_id, date_time, settings, tiny, session_dir, record, misc):
    manifest = record['manifest']
    dumps = lambda value: json.dumps(value, cls=ut.NumpyEncoder)
    return {
        'ID': session_id, 'date_time': date_time,
        **session.version_labels(settings['prompt_version'], tiny.key),
        'tier': settings['tier'], 'tiny_model': tiny.key, 'tiny_revision': tiny.revision, 'task': settings['task'],
        'provider': settings['provider'], 'model': settings['model'], 'llm_kwargs': dumps(settings['llm_kwargs']),
        'status': record['status'], 'n_attempts': record['n_attempts'],
        'tensors_edited': dumps([t['name'] for t in manifest['tensors']]),
        'entries_changed': manifest['entries_changed'],
        'relative_change': dumps({t['name']: t['relative_change'] for t in manifest['tensors']}),
        'total_cost': float(record['total_cost']), 'token_list': dumps(record['token_list']),
        'wall_s': float(record['wall_s']), 'session_dir': _project_relative(session_dir),
        'checkpoint_dir': _project_relative(record['checkpoint_dir']) if record['checkpoint_dir'] else '',
        'misc': dumps(misc),
    }


def _project_relative(path):
    # A checkpoint may be on another drive, where no relative path exists; it stays absolute.
    try:
        return os.path.relpath(path, cfg.REPO_ROOT)
    except ValueError:
        return path


def _read_master():
    if not os.path.exists(cfg.MASTER_CSV):
        return pl.DataFrame(schema=MASTER_SCHEMA)
    with open(cfg.MASTER_CSV, encoding='utf-8') as f:
        columns = f.readline().strip().split(',')
    rows = pl.read_csv(cfg.MASTER_CSV, schema={name: MASTER_SCHEMA[name] for name in columns})
    missing = [name for name in MASTER_SCHEMA if name not in columns]
    rows = rows.with_columns(pl.lit(COLUMNS_ADDED_LATER[name], dtype=MASTER_SCHEMA[name]).alias(name) for name in missing)
    return rows.select(list(MASTER_SCHEMA))


def _append_master_row(row):
    """Read, concat, write. Call it under ut._save_lock. That lock is per process: two processes
    appending at once can lose a row."""
    updated = pl.concat([_read_master(), pl.DataFrame({key: [value] for key, value in row.items()}, schema=MASTER_SCHEMA)])
    os.makedirs(os.path.dirname(cfg.MASTER_CSV), exist_ok=True)
    ut.write_csv_with_retry(updated, cfg.MASTER_CSV)
