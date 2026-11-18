"""Run a served tiny model over one evaluation set, grade the answers and save a row of results.

Where it sits in the flow: below scoring.score_tiny_model, which registers the tiny model with the
local server and calls eval_benchmark_async once per pass. Nothing here knows about edits; a model is
a key of query.LOCAL_MODEL_CONFIG.

Reads data/eval/<evaluation set>_Benchmark.jsonl. Writes, under config.SCORES_DIR,
Master_LLM_<evaluation set>_Results.csv (one row per pass) and
<evaluation set>_model_answers/<row ID>_<evaluation set>_model_answers.txt (the full answers).
"""
import asyncio
import datetime
import json
import os

import numpy as np
import polars as pl
from openai import APIStatusError

from wspmathbench import arithmetic
from wspmathbench import config as cfg
from wspmathbench.llm import query as qry
from wspmathbench.llm import registry
from wspmathbench.llm import utils as ut

BOXED_GRADING, SYMBOLIC_BOXED_GRADING, FIRST_NUMBER_GRADING = 2, 3, 'first_number'
BOXED_SYSTEM_PROMPT = 2

# The evaluation sets. Each has one grading rule and one way of giving the instruction, and a row
# records both in its misc under the codes below.
#   question_column     the column of the .jsonl file holding the question; 'answer' holds the reference
#   grading_type        2: the last \boxed{} answer as an integer; 3: the last \boxed{} answer, checked
#                       for mathematical equivalence; 'first_number': the first number in the reply
#   worker_prompt_type  2: BOXED_INSTRUCTION is sent as the system message; None: no system message
#   failure_answer      what is recorded for a question with no usable answer. It never equals a
#                       reference answer, so it always scores wrong.
#   integer_answers     the reference answers are integers and are compared as integers
EVAL_SETS = {
    'GSM8K_500': {'question_column': 'question', 'grading_type': BOXED_GRADING,
                  'worker_prompt_type': BOXED_SYSTEM_PROMPT, 'failure_answer': 0, 'integer_answers': True},
    'MATH500': {'question_column': 'problem', 'grading_type': SYMBOLIC_BOXED_GRADING,
                'worker_prompt_type': BOXED_SYSTEM_PROMPT, 'failure_answer': 'NA', 'integer_answers': False},
    'BasicArithmetic': {'question_column': 'problem', 'grading_type': FIRST_NUMBER_GRADING,
                        'worker_prompt_type': None, 'failure_answer': -1, 'integer_answers': True},
}

# Default llm_kwargs of a run, merged under the caller's. The merged dict is saved in the row's
# llm_kwargs column, so a change here changes the recorded metadata of every later row.
EVAL_LLM_DEFAULTS = {'thinking': True, 'max_tokens': None, 'stream': False, 'verbose': False}

# The model that reads the final answer out of a reply when no fixed rule can (see Grading below).
ANSWER_HELPER = {'model': 'gpt-5.6-luna', 'reasoning_effort': 'low'}
ANSWER_HELPER_RETRIES = 100

# Marks, in a model-answers file, a reply that was cut at its output limit before any answer text and
# is saved as its reasoning text (save_truncated).
TRUNCATED_PREFIX = '[TRUNCATED, reasoning only] '

MASTER_SCHEMA = {
    'ID': pl.Int64, 'date_time': pl.String, 'provider': pl.String, 'model': pl.String,
    'llm_kwargs': pl.String, 'score': pl.Float64, 'model_answers': pl.String, 'total_cost': pl.Float64,
    'cost_list': pl.String, 'answer_cost_list': pl.String, 'token_list': pl.String, 'misc': pl.String,
}

_BOXED_HELPER_PROMPT = (
    "You are a helper that locates a mathematical answer given in a larger passage of text. "
    "You are going to be presented with some text that is a response to a mathematical question. "
    r"The solution to the question is a positive integer number, which should be given inside of \boxed{} "
    r"within the presented text. Your task is to return just the integer answer that was inside "
    r"of \boxed{} from the given text. "
    r"If the text contains multiple instances of \boxed{} with answers in them, "
    r"return only the answer from the last one. "
    r"If the text does not contain any instances of \boxed{} with an answer inside, return 0. "
    r"If the text contains a clear answer that is not inside of \boxed{}, return 0, even if that answer "
    r"is bolded, underlined, or otherwise emphasized. "
    r"The answer MUST be inside of \boxed{} for it to be counted. "
    "If the text is incomplete and does not give a final answer, return 0. "
    "If the text is unclear on what its final answer is, return 0. "
    "Omit any commas or decimals from the answer in your extracted response. "
    r"Return only the plain integer, with no \boxed{}, braces, backslashes, formatting, explanation, "
    "or other text."
)


def _symbolic_boxed_helper_prompt(no_answer):
    return (
        "You are helper that locates a mathematical answer given in a larger passage of text."
        "You are going to be presented with some text that is a response to a mathematical question."
        r"The answer to should be given inside of \boxed{}"
        "within the presented text. Your task is to return the answer from the given text. "
        r"If the text contains multiple instances of \boxed{} with answers in them, "
        r"return only the answer from the last one. "
        r"If the text does not contain any instances of \boxed{} with an answer inside, "
        f"return {no_answer}. "
        r"If the text does not contain an answer inside of \boxed{}, "
        f"return {no_answer}. "
        r"If the text does contain a clear answer, but it is not inside of \boxed{}, "
        f"return {no_answer}. "
        f"If the text is incomplete and does not give a final answer, return {no_answer}. "
        f"If the text is unclear on what its final answer is, return {no_answer}. "
        "Do not include anything else in your response, just the plain formatted answer that "
        f"you have extracted from the given text or {no_answer}"
    )


def eval_set_file(eval_set):
    return os.path.join(cfg.EVAL_DIR, f'{eval_set}_Benchmark.jsonl')


def master_file(eval_set):
    return os.path.join(cfg.SCORES_DIR, f'Master_LLM_{eval_set}_Results.csv')


def model_answers_file(eval_set, row_id):
    return os.path.join(cfg.SCORES_DIR, f'{eval_set}_model_answers', f'{row_id}_{eval_set}_model_answers.txt')


def load_eval_set(eval_set):
    """(questions, reference answers) of an evaluation set, in file order. The reference answers are
    integers where the set's answers are integers, and strings otherwise."""
    spec = _eval_spec(eval_set)
    rows = pl.read_ndjson(eval_set_file(eval_set))
    answers = rows['answer'].to_list()
    if spec['integer_answers']:
        answers = [int(answer) for answer in answers]
    return rows[spec['question_column']].to_list(), answers


def _eval_spec(eval_set):
    if eval_set not in EVAL_SETS:
        raise ValueError(f'`eval_set` must be one of {list(EVAL_SETS)}, got {eval_set!r}.')
    return EVAL_SETS[eval_set]


def eval_benchmark(**arguments):
    """Synchronous eval_benchmark_async (same arguments, same return value)."""
    return asyncio.run(eval_benchmark_async(**arguments))


async def eval_benchmark_async(
        eval_set='GSM8K_500', max_workers=6, provider='local', model=None, llm_kwargs=None,
        worker_prompt=True, Notes=None, Semaphore=None, max_worker_retries=3,
        smoketest=False, salvage_truncated=False, retry_wait='strat1', live_status=None,
        misc_extra=None, save_truncated=False):
    """One pass of one model over one evaluation set: ask every question, grade the answers and
    append a row to the set's master CSV. Returns the fraction of questions answered correctly.

    Parameters
    ----------
    eval_set : 'GSM8K_500', 'MATH500' or 'BasicArithmetic' (a key of EVAL_SETS)
    max_workers : questions in flight at once (the size of the semaphore made when `Semaphore` is None)
    provider, model : provider is 'local'; model is a key of query.LOCAL_MODEL_CONFIG, such as the
        'wsp-<tiny model>-base' or 'wsp-<tiny model>-<session ID>' that query.wsp_local_model registers
    llm_kwargs : merged over EVAL_LLM_DEFAULTS and passed to the query function for every question.
        The merged dict is printed and saved in the row's llm_kwargs column.
    worker_prompt : True sends the set's instruction (below); False sends no system message
    Notes : free text saved under misc['Notes']
    Semaphore : an asyncio.Semaphore shared by concurrent runs to cap their combined concurrency
    max_worker_retries : attempts per question. Every exception is retried.
    smoketest : ask the first question only and save nothing
    salvage_truncated : when a reply hit its output limit with no answer text, grade its reasoning
        text instead of an empty answer
    save_truncated : when a reply hit its output limit with no answer text, save its reasoning text
        in the model-answers file, prefixed with TRUNCATED_PREFIX, instead of an empty string. This
        does not change what is graded.
    retry_wait : 'strat1' waits 5, 10, 20, ... seconds between attempts; 'strat2' waits a jittered
        3 to 10 seconds that grows slowly with the attempt number
    live_status : None, False, True, 'tokens' or 'chars'. The live status footer (utils.live_status);
        None means utils.LIVE_STATUS_DEFAULT. Display only.
    misc_extra : a dict merged into the row's misc as given

    The instruction
    ---------------
    GSM8K_500 and MATH500: "Please reason step by step, and put your final answer within \\boxed{}."
    is sent as the system message, and the user message is the question alone (worker_prompt_type 2).
    BasicArithmetic: no instruction; the question is sent exactly as written (worker_prompt_type None).

    Grading
    -------
    GSM8K_500 (grading_type 2): a reply that is nothing but digits is the answer. An empty reply is
        graded 0. Any other reply goes to the answer helper (ANSWER_HELPER, an OpenAI model), which
        returns the integer inside the last \\boxed{}, or 0 when there is none, when the answer is not
        inside \\boxed{}, or when the text is incomplete or unclear. The integer is compared with the
        reference answer. No reference answer is 0, so 0 always scores wrong.
    MATH500 (grading_type 3): a reply that is nothing but digits is the answer. An empty reply is
        graded 'NA'. Otherwise the content of the last \\boxed{} is the answer, read by a fixed rule;
        only a reply with no readable \\boxed{} goes to the answer helper, which returns the boxed
        answer or 'NA'. The answer is compared with the reference answer by math-verify
        (math_grader.MathVerifyGrader, 10 seconds per answer); 'NA' always scores wrong.
    BasicArithmetic ('first_number'): the first number in the reply, written in digits or in words
        (arithmetic.first_number), compared with the reference answer as an integer. A reply with no
        number is recorded as -1. No helper is called.
    A question that fails every attempt is recorded with the set's failure answer (0, 'NA' or -1), no
    answer text and no tokens.

    Saved
    -----
    One row of Master_LLM_<eval_set>_Results.csv: ID, date_time, provider, model, llm_kwargs, score,
    model_answers (the graded answers, in file order), total_cost, cost_list (answer, helper and lost
    cost per question), answer_cost_list (the answering call alone), token_list ((prompt, cached,
    completion, reasoning, total) per question) and misc: worker_prompt, worker_prompt_type,
    grading_type, salvage_truncated, save_truncated, server_providers when any was reported, Notes,
    correct_list (1 or 0 per question, BasicArithmetic only) and the keys of misc_extra. The full
    replies go to <eval_set>_model_answers/<ID>_<eval_set>_model_answers.txt.
    """
    spec = _eval_spec(eval_set)
    if provider != 'local':
        raise ValueError(f"`provider` must be 'local', got {provider!r}.")
    if model not in qry.LOCAL_MODEL_CONFIG:
        raise ValueError(f'`model` must be one of {list(qry.LOCAL_MODEL_CONFIG)}, got {model!r}.')
    if Semaphore is None:
        Semaphore = asyncio.Semaphore(max_workers)
    datetime_str = str(datetime.datetime.now())

    llm_kwargs = {**EVAL_LLM_DEFAULTS, **(llm_kwargs or {})}
    registry.check_thinking_arm(llm_kwargs.get('base_model', model), llm_kwargs.get('thinking'))
    print(f'llm_kwargs: {llm_kwargs}')

    questions, answers = load_eval_set(eval_set)
    if smoketest:
        questions, answers = questions[:1], answers[:1]

    grading_type, failure_answer = spec['grading_type'], spec['failure_answer']
    first_number_grading = grading_type == FIRST_NUMBER_GRADING
    system_prompt = None
    if worker_prompt and spec['worker_prompt_type'] == BOXED_SYSTEM_PROMPT:
        system_prompt = cfg.BOXED_INSTRUCTION

    async def extract_answer(answer):
        """(the answer read from a reply that is not all digits, the helper's cost)."""
        if first_number_grading:
            number = arithmetic.first_number(answer or '')
            return str(failure_answer if number is None else number), 0.0
        if not answer or not answer.strip():
            return str(failure_answer), 0.0
        if grading_type == BOXED_GRADING:
            helper_prompt = _BOXED_HELPER_PROMPT
        else:
            helper_prompt = _symbolic_boxed_helper_prompt(failure_answer)
            if (r'\boxed{' in answer and '}' in answer) or (r'\fbox{' in answer and '}' in answer):
                boxed = ut.remove_boxed(ut.last_boxed_only_string(answer))
                if boxed:
                    return boxed, 0

        helper_full, _, _ = await registry.retry_llm_call(
            lambda: qry.openai_query_async(
                prompt=answer, system_prompt=helper_prompt, verbose=False,
                prompt_cache_key='answer_helper_instructions', **ANSWER_HELPER),
            max_retries=ANSWER_HELPER_RETRIES, label='answer helper', retry_wait='strat2')
        helper_answer = helper_full['text']
        if grading_type == BOXED_GRADING:
            if r'\boxed{' in answer and '}' in answer:  # the helper may return the \boxed{} wrapper as well
                helper_answer = helper_answer.replace('boxed{', '').replace('}', '')
            if type(ut.safer_int(helper_answer)) != int:
                print('The answer helper returned no integer; the answer is recorded as 0.')
                helper_answer = '0'
        return helper_answer, helper_full['cost_estimate']

    def retry_wait_s(attempt):
        return 2 ** attempt * 5 if retry_wait == 'strat1' else ut.wait_strat2_helper(attempt)

    async def worker(question, i):
        failed = (failure_answer, 0.0, 0, 0, None, '')
        async with Semaphore:
            print(f'index: {i}')
            server_provider = None
            lost_cost = 0.0
            for attempt in range(max_worker_retries):
                ut.live_status_attempt(attempt)
                context = f'{eval_set} index {i}, attempt {attempt + 1}'
                try:
                    full = await registry.query_base_model(provider, model, question,
                                                           system_prompt=system_prompt, **llm_kwargs)
                    tokens = (full['prompt_tokens'], full['cached_tokens'], full['completion_tokens'],
                              full['reasoning_tokens'], full['total_tokens'])
                    answer_full = full['text'] or ''
                    truncated = full.get('incomplete_reason') == 'max_tokens'
                    server_provider = full.get('server_provider')
                    # What the model-answers file gets: the graded text, except that with save_truncated a
                    # reply cut before any answer text is saved as its reasoning, marked with the prefix.
                    cut_reasoning = None
                    if save_truncated and truncated and not answer_full.strip():
                        cut_reasoning = (full.get('reasoning_text') or '').strip() or None
                    if not answer_full.strip():
                        if truncated and salvage_truncated:
                            answer_full = (full.get('reasoning_text') or '').strip() or answer_full
                        elif not truncated and full['text'] is None:
                            # A completed reply with no text is a fault of the server: try again.
                            lost_cost += float(full['cost_estimate'])
                            raise ValueError(f'index {i}: {model} returned no text')
                    helper_cost = 0
                    if first_number_grading or not answer_full.isdigit():
                        answer, helper_cost = await extract_answer(answer_full)
                    else:
                        answer = answer_full
                    cost = round(full['cost_estimate'] + helper_cost + lost_cost, 9)
                    answer_cost = round(full['cost_estimate'], 9)
                    print(f'index: {i}, model answer: {answer}, correct answer: {answers[i]}, '
                          f'server_provider: {server_provider}, cost: {cost}, tokens: {tokens}.')
                    saved = answer_full if cut_reasoning is None else TRUNCATED_PREFIX + cut_reasoning
                    return answer, cost, answer_cost, tokens, server_provider, saved

                except APIStatusError as exc:
                    if attempt < max_worker_retries - 1:
                        wait = retry_wait_s(attempt)
                        print(f'index {i}: status {exc.status_code}{ut.exc_location(exc)} on attempt '
                              f'{attempt + 1}, retrying in {wait}s')
                        await asyncio.sleep(wait)
                        continue
                    print(f'index {i}: no answer after {max_worker_retries} attempts (status '
                          f'{exc.status_code}): {ut.report_exc(exc, context)}')
                    return (failure_answer, lost_cost, 0, 0, server_provider, '')

                except Exception as exc:
                    if attempt < max_worker_retries - 1:
                        wait = retry_wait_s(attempt)
                        print(f'index {i}: {ut.report_exc(exc, context)} on attempt {attempt + 1}, '
                              f'retrying in {wait}s.')
                        await asyncio.sleep(wait)
                        continue
                    print(f'index {i}: no answer after {max_worker_retries} attempts: '
                          f'{ut.report_exc(exc, context)}')
                    return (failure_answer, lost_cost, 0, 0, server_provider, '')
        return failed

    # live_status_task tags each worker with its question index, so the footer can show it.
    with ut.live_status(eval_set, len(questions), live_status) as status:
        tasks = [ut.live_status_task(status, i, worker(question, i)) for i, question in enumerate(questions)]
        results = await asyncio.gather(*tasks)

    if grading_type == BOXED_GRADING:
        model_answers = [ut.safer_int(r[0]) for r in results]
    elif first_number_grading:
        model_answers = [int(r[0]) for r in results]
    else:
        model_answers = [r[0] for r in results]
    cost_list = [r[1] for r in results]
    answer_cost_list = [r[2] for r in results]
    token_list = [r[3] for r in results]
    server_providers = [r[4] for r in results]
    model_answers_full = [r[5] for r in results]
    if len(set(server_providers)) == 1:
        server_providers = list(set(server_providers))
    total_cost = np.sum(cost_list)

    correct_list = None
    if grading_type == BOXED_GRADING:
        num_correct = int((np.array(model_answers) == np.array(answers)).sum())
    elif first_number_grading:
        correct_list = [int(model_answer == answer) for model_answer, answer in zip(model_answers, answers)]
        num_correct = sum(correct_list)
    else:
        from wspmathbench.llm.math_grader import MathVerifyGrader
        with MathVerifyGrader(timeout=10) as grader:
            num_correct = sum(1 for model_answer, answer in zip(model_answers, answers)
                              if model_answer != failure_answer and grader.grade(answer, model_answer) == 'correct')
    score = num_correct / len(answers)
    print(f'Score: {score}')
    print(f'total_cost: {total_cost}')

    misc = run_misc(eval_set, worker_prompt=worker_prompt, salvage_truncated=salvage_truncated,
                    save_truncated=save_truncated)
    if any(server_provider is not None for server_provider in server_providers):
        misc['server_providers'] = server_providers
    if Notes is not None:
        misc['Notes'] = Notes
    if correct_list is not None:
        misc['correct_list'] = correct_list
    if misc_extra:
        misc.update(misc_extra)

    if not smoketest:
        results = (provider, model, llm_kwargs, score, model_answers, total_cost, cost_list,
                   answer_cost_list, token_list, misc, model_answers_full)
        async with ut._save_lock:
            save_eval_benchmark_results(eval_set, results, datetime_str)
    return score


def run_misc(eval_set, worker_prompt=True, salvage_truncated=False, save_truncated=False):
    """The settings every row of an evaluation set records in its misc. A set with integer-boxed or
    symbolic-boxed grading records the code (2 or 3); BasicArithmetic records 'first_number'."""
    spec = _eval_spec(eval_set)
    return {'worker_prompt': worker_prompt, 'worker_prompt_type': spec['worker_prompt_type'],
            'grading_type': spec['grading_type'], 'salvage_truncated': salvage_truncated,
            'save_truncated': save_truncated}


def read_master(eval_set):
    """The master CSV of an evaluation set, or an empty table with its columns if none exists yet."""
    path = master_file(eval_set)
    if not os.path.exists(path):
        return pl.DataFrame(schema=MASTER_SCHEMA)
    return pl.read_csv(path, schema=MASTER_SCHEMA)


def save_eval_benchmark_results(eval_set, results, datetime_str):
    """Append one row to the evaluation set's master CSV (created on first use) and write the full
    answers next to it. Returns the row's ID, one more than the last row's. Call it under
    utils._save_lock: that lock is per process, so two processes saving at once can lose a row."""
    (provider, model, llm_kwargs, score, model_answers, total_cost, cost_list, answer_cost_list,
     token_list, misc, model_answers_full) = results
    rows = read_master(eval_set)
    new_id = 0 if rows.is_empty() else rows[-1, 0] + 1
    dumps = lambda value: json.dumps(value, cls=ut.NumpyEncoder)
    new_row = pl.DataFrame({
        'ID': [new_id], 'date_time': [datetime_str], 'provider': [provider], 'model': [model],
        'llm_kwargs': [dumps(llm_kwargs)], 'score': [float(score)], 'model_answers': [dumps(model_answers)],
        'total_cost': [float(total_cost)], 'cost_list': [dumps(cost_list)],
        'answer_cost_list': [dumps(answer_cost_list)], 'token_list': [dumps(token_list)],
        'misc': [dumps(misc)],
    }, schema=MASTER_SCHEMA)
    os.makedirs(cfg.SCORES_DIR, exist_ok=True)
    ut.write_csv_with_retry(pl.concat([rows, new_row]), master_file(eval_set))
    print(f'Saved row {new_id} of {os.path.basename(master_file(eval_set))}')

    # The answers file is the Python text of the list of replies, in file order.
    answers_path = model_answers_file(eval_set, new_id)
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    with open(answers_path, 'w', encoding='utf-8') as f:
        f.write(str(model_answers_full))
    print(f'Saved model answers: {answers_path}')
    return new_id
