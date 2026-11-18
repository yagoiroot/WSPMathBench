"""Score a tiny model, unedited or edited, on its held-out evaluation sets.

Where it sits in the flow: after the edit sessions. score_tiny_model registers the model with the
local server (query.wsp_local_model), starts that server, and runs benchmarks.eval_benchmark_async
once per pass and evaluation set. scripts/score_tiny_model.py is its command line.

Reads results/Master_WSP_Edits.csv (to recognise a session that saved no edit) and the checkpoint's
edits.json. Writes the rows and answer files described in benchmarks.py.
"""
import asyncio
import datetime
import json
import os

import polars as pl

from wspmathbench import config as cfg
from wspmathbench.llm import benchmarks as bm
from wspmathbench.llm import query as qry

# A math model scored with fewer new tokens than this is almost certainly a mistake (the limit of
# TinyStories-33M, 16, carried over to another model).
_MIN_MATH_MAX_TOKENS = 1000
_NO_EDIT_NOTES = {
    None: '',
    'program_raised': (" NO MODEL WAS RUN: the session's program raised and saved no edit, so every "
                       "question is recorded as unanswered and the score is 0."),
    'ran_no_edit': (" The session's program ran without error and saved no edit, so the UNEDITED "
                    "model was run."),
}


def _wsp_no_edit_session(session_id):
    """Whether a session ended with no edit, from its row of Master_WSP_Edits.csv.

    Returns (kind, tiny model, task, exception). kind is 'program_raised' for an 'empty_edit' row
    whose misc records an exception (a oneshot_code program that raised before saving anything),
    'ran_no_edit' for an 'empty_edit' row without one (the reply was applied or run without error
    and changed nothing), and None for every other row and for a session with no row; the other
    three are then None too."""
    if not os.path.exists(cfg.MASTER_CSV):
        return None, None, None, None
    rows = pl.read_csv(cfg.MASTER_CSV, infer_schema_length=None).filter(pl.col('ID') == session_id)
    if rows.is_empty() or rows[0, 'status'] != 'empty_edit':
        return None, None, None, None
    row = rows.row(0, named=True)
    exception = json.loads(row['misc'] or '{}').get('exception')
    return ('program_raised' if exception else 'ran_no_edit'), row['tiny_model'], row['task'], exception


def _wsp_zero_score_rows(model, datasets, repeats, info, run_notes, llm_kwargs, smoketest=False):
    """Append `repeats` rows with score 0 to each evaluation set's master CSV, for a session whose
    program raised: there is no edited model, so nothing is run. A row has the shape of a real one,
    with every question recorded the way eval_benchmark_async records a question that got no answer
    (0 for GSM8K_500, 'NA' for MATH500, -1 for BasicArithmetic), an empty answer text, no cost and
    no tokens. misc['not_run'] is True, and misc['wsp'] carries 'no_edit': 'program_raised' and the
    exception. With smoketest nothing is saved."""
    for eval_set in datasets:
        failure_answer = bm.EVAL_SETS[eval_set]['failure_answer']
        n = len(bm.load_eval_set(eval_set)[0])
        misc = bm.run_misc(eval_set, worker_prompt=True, salvage_truncated=False, save_truncated=True)
        misc['Notes'] = run_notes
        if misc['grading_type'] == bm.FIRST_NUMBER_GRADING:
            misc['correct_list'] = [0] * n
        misc.update({'wsp': info, 'not_run': True})
        for i in range(repeats):
            results = ('local', model, llm_kwargs, 0.0, [failure_answer] * n, 0.0, [0.0] * n, [0.0] * n,
                       [[0, 0, 0, 0, 0] for _ in range(n)], misc, [''] * n)
            if smoketest:
                print(f'{model} {eval_set}: smoketest, a score-0 row of {n} questions would be saved')
            else:
                bm.save_eval_benchmark_results(eval_set, results, str(datetime.datetime.now()))
                print(f'{model} {eval_set}: saved score-0 row {i + 1} of {repeats} ({n} questions, nothing run)')


def score_tiny_model(tiny_model=None, session_id=None, datasets=None, repeats=3, max_tokens=None,
                     max_workers=None, notes=None, smoketest=False):
    """Score a tiny model on its held-out evaluation sets, on the local server.

    tiny_model : a key of config.TINY_MODEL_REGISTRY ('Qwen3.5-0.8B', 'MobileLLM-R1-950M',
        'LFM2.5-350M', 'gemma-3-270m-it', 'TinyStories-33M'). With session_id=None the UNEDITED model
        is scored; do that once per tiny model, since every edit of it is compared with the same
        unedited rows.
    session_id : the ID of an edit session whose checkpoint was built. That EDITED checkpoint is
        scored; it names its own tiny model, so `tiny_model` is not needed.
        A session that saved no edit (status 'empty_edit' in Master_WSP_Edits.csv) has no checkpoint
        and is handled by kind. Both kinds record their rows under the session's own key,
        'wsp-<tiny_model>-<session ID>':
          - its program raised (the row's misc holds an exception): no model is run. `repeats` rows
            with score 0 are appended per evaluation set, every question recorded as unanswered.
          - it ran without error and changed nothing: the UNEDITED model is served and run as usual.
        misc['wsp']['no_edit'] says which ('program_raised' or 'ran_no_edit').
    datasets : evaluation sets (keys of benchmarks.EVAL_SETS); None means the task's own
        ('GSM8K_500' and 'MATH500' for the math task, 'BasicArithmetic' for TinyStories-33M).
    repeats : passes per evaluation set. Every protocol samples, so passes differ (only a little for
        LFM2.5-350M and TinyStories-33M, which run at temperature 0.1).
    max_tokens : new tokens per answer; None means the protocol's own limit (32,768 for
        Qwen3.5-0.8B, 30,000 for the other math models). TinyStories-33M always runs with its
        protocol's 16. An edited model and its unedited model must be scored with the same value:
        the limit decides the score of a model that runs to it.
    max_workers : questions in flight; None means 128 (the servers' --max-num-seqs), or 16 for the
        transformers server, which answers one request at a time.
    notes : free text appended to the Notes of every row.
    smoketest : ask the first question of each evaluation set once per pass and save nothing.

    The sampling settings are the tiny model's protocol (config.PROTOCOLS). The instruction and the
    grading rule are the evaluation set's (benchmarks.eval_benchmark_async): the \\boxed{} instruction
    as the system message with the last \\boxed{} answer graded for GSM8K_500 and MATH500, and the
    first number of the continuation for BasicArithmetic. GSM8K_500 and MATH500 can call the answer
    helper (an OpenAI model, so OPENAI_API_KEY must be set and a pass has a small API cost);
    BasicArithmetic costs nothing. A reply cut at the token limit before any answer text is graded as
    unanswered, and its reasoning is saved in the answers file with benchmarks.TRUNCATED_PREFIX.

    Rows go to results/scores/Master_LLM_<evaluation set>_Results.csv with provider 'local' and model
    'wsp-<tiny_model>-base' or 'wsp-<tiny_model>-<session ID>'. misc['wsp'] holds the session ID,
    tiny model, task and protocol. A per-question comparison of edited against unedited is made from
    each row's model_answers; BasicArithmetic rows also carry misc['correct_list'].
    """
    # A session that saved no edit has no checkpoint. Its master row says which kind it is and
    # names the tiny model.
    no_edit, no_edit_exception = None, None
    if session_id is not None:
        no_edit, no_edit_tiny, no_edit_task, no_edit_exception = _wsp_no_edit_session(session_id)
        if no_edit:
            if tiny_model and tiny_model != no_edit_tiny:
                raise ValueError(f'Session {session_id} is for {no_edit_tiny}, not {tiny_model}.')
            tiny_model = no_edit_tiny

    if tiny_model == 'TinyStories-33M':
        max_tokens = cfg.PROTOCOLS['tinystories_33m']['sampling']['max_tokens']
    if no_edit:
        # The unedited model's entry, registered under the session's own key, so that the rows are
        # the session's while the weights served are the unedited ones.
        base = qry.wsp_local_model(tiny_model=tiny_model, max_tokens=max_tokens)
        base_info = qry.LOCAL_MODEL_CONFIG[base]['wsp']
        if no_edit_task != base_info['task']:
            raise ValueError(f"Session {session_id} is for task {no_edit_task!r}, but the unedited "
                             f"{tiny_model} is served for {base_info['task']!r}.")
        model = f'wsp-{tiny_model}-{session_id}'
        qry.LOCAL_MODEL_CONFIG[model] = {
            **qry.LOCAL_MODEL_CONFIG[base],
            'wsp': {**base_info, 'session_id': session_id, 'no_edit': no_edit,
                    **({'exception': str(no_edit_exception)[:500]} if no_edit == 'program_raised' else {})}}
    else:
        model = qry.wsp_local_model(tiny_model=tiny_model, session_id=session_id, max_tokens=max_tokens)

    served = qry.LOCAL_MODEL_CONFIG[model]
    info = served['wsp']
    max_tokens = served['max_tokens']
    if info['tiny_model'] != 'TinyStories-33M' and max_tokens < _MIN_MATH_MAX_TOKENS:
        raise ValueError(f'max_tokens={max_tokens} is too low for {info["tiny_model"]}: its answers '
                         f'need room to reason. Leave it unset for the protocol\'s own limit.')
    datasets = list(cfg.TASKS[info['task']]['eval_sets']) if datasets is None else list(datasets)
    for eval_set in datasets:
        if eval_set not in bm.EVAL_SETS:
            raise ValueError(f'`datasets` must be among {list(bm.EVAL_SETS)}, got {eval_set!r}.')
    if max_workers is None:
        max_workers = 16 if served['engine'] == 'transformers' else 128
    stream = False

    run_notes = (f"WSP {'unedited' if info['session_id'] is None else 'session ' + str(info['session_id'])} "
                 f"{info['tiny_model']}; protocol {info['protocol']}; max_tokens={max_tokens}; max_workers={max_workers}; "
                 f"stream={stream}; server_args: {served['server_args']}; salvage_truncated=False."
                 + _NO_EDIT_NOTES[no_edit] + (f' {notes}' if notes else ''))
    print(f'{model}: {datasets}, {repeats} pass(es) each, max_tokens {max_tokens}, max_workers {max_workers}')

    if no_edit == 'program_raised':  # scored 0 without running anything
        _wsp_zero_score_rows(model=model, datasets=datasets, repeats=repeats, info=info, run_notes=run_notes,
                             llm_kwargs={**bm.EVAL_LLM_DEFAULTS, 'stream': stream, 'max_tokens': max_tokens},
                             smoketest=smoketest)
        return

    # Stops whatever else holds the server port, launches this model and waits until it answers.
    qry.switch_local_server(model=model)

    async def run_passes():
        # One event loop for every pass. The local client keeps its connections to the server open
        # between requests, and a connection opened in one asyncio.run() fails when the next
        # asyncio.run() tries to reuse it, which costs every worker a retry.
        for eval_set in datasets:
            for _ in range(repeats):
                await bm.eval_benchmark_async(
                    eval_set=eval_set, max_workers=max_workers, provider='local', model=model,
                    max_worker_retries=3, salvage_truncated=False, save_truncated=True,
                    live_status='tokens', worker_prompt=True,
                    llm_kwargs={'stream': stream, 'max_tokens': max_tokens},
                    Notes=run_notes, misc_extra={'wsp': info}, smoketest=smoketest)

    asyncio.run(run_passes())
