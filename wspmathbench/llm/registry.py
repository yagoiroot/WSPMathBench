"""Which models can be queried, the routing of a query to its provider's function, and the retry
ladder around a call.

Where it sits in the flow: the edit sessions and the scoring code call query_base_model (usually
inside retry_llm_call) and never a provider function directly.

Reads and writes nothing.
"""
import asyncio

from anthropic import APIStatusError as AnthropicAPIStatusError
from openai import APIStatusError

from wspmathbench.llm import query as qry
from wspmathbench.llm import utils as ut

# Provider -> its query function and the models it serves. The primary LLMs of the paper, plus
# gpt-5-nano, which is cheap enough for trial runs, and gpt-5.6-luna, which the scoring code uses
# to read a final answer out of a reply.
#
# To add a model of a provider that is already here: add its name to 'models' below and its entry
# (prices, limits, how its reasoning is controlled) to MODEL_CONFIG inside that provider's
# function in query.py.
# To add a provider: write an async function in query.py that takes (prompt, model, **settings)
# and returns the response contract described at the top of that module, and add a line here.
BASE_MODEL_REGISTRY = {
    'OpenAI':    {'query': qry.openai_query_async,
                  'models': ['gpt-6.1-sol', 'gpt-6-astra', 'gpt-6-astra-pro', 'gpt-5-nano',
                             'gpt-5.6-luna']},
    'Anthropic': {'query': qry.anthropic_query_async,
                  'models': ['claude-fable-5-2']},
    'Google':    {'query': qry.google_query_async,
                  'models': ['gemini-4-carbon']},
    'DeepSeek':  {'query': qry.deepseek_query_async,
                  'models': ['deepseek-v4.1-flash-1017']},
    'Kimi':      {'query': qry.kimi_query_async,
                  'models': ['kimi-k3.1-pro']},
    'local':     {'query': qry.local_query_async,
                  # a live view, not a copy: qry.wsp_local_model registers entries at run time
                  'models': qry.LOCAL_MODEL_CONFIG.keys()},
}

# The provider SDKs raise unrelated APIStatusError classes; both carry .status_code.
API_STATUS_ERRORS = (APIStatusError, AnthropicAPIStatusError)

# Models that cannot run without thinking (their entries in query.py: supports_disable False for
# Anthropic, thinking_control 'effort' for Kimi). Checked when a run starts, outside any retry
# ladder, so that a thinking-off configuration raises instead of being retried into empty answers.
NO_THINKING_OFF_MODELS = ['claude-fable-5-2', 'kimi-k3.1-pro']

def check_thinking_arm(base_model, thinking):
    if base_model in NO_THINKING_OFF_MODELS and thinking in (False, 'disabled'):
        raise ValueError(f"'{base_model}' cannot run without thinking (thinking={thinking!r}); "
                         f"use thinking=None/'adaptive' and set reasoning_effort instead.")

async def query_base_model(base_provider, base_model, prompt, **kwargs):
    """Route one query to the registered function of `base_provider`.

    kwargs are passed through as given: each caller supplies exactly the settings it wants for
    that provider, so that nothing is decided here."""
    entry = BASE_MODEL_REGISTRY.get(base_provider)
    if entry is None:
        raise ValueError(f'`base_provider` must be one of {list(BASE_MODEL_REGISTRY)}')
    if base_model not in entry['models']:
        raise ValueError(f"Model '{base_model}' is not registered for provider "
                         f"'{base_provider}'. Registered: {list(entry['models'])}")
    fn = entry['query']
    if asyncio.iscoroutinefunction(fn):
        return await fn(prompt=prompt, model=base_model, **kwargs)
    return fn(prompt=prompt, model=base_model, **kwargs)

async def retry_llm_call(call, max_retries=3, label='LLM call', retry_wait='strat1'):
    """The retry ladder around one LLM call. An API status error and any other exception are both
    retried, and a reply whose 'text' is None counts as a failed attempt whose billed cost is added
    to lost_cost.

    `call` is a zero-argument callable returning a fresh coroutine that resolves to a response
    contract dict.
    retry_wait: 'strat1' waits 5, 10, 20 s between attempts; 'strat2' waits a few seconds with
    jitter (ut.wait_strat2_helper).

    Returns (response or None, lost_cost, last_error)."""
    lost_cost = 0.0
    last_error = None
    for attempt in range(max_retries):
        try:
            full = await call()
            if full['text'] is None:
                lost_cost += float(full.get('cost_estimate') or 0)
                raise ValueError(f'returned no text (provider={full.get("server_provider")}, '
                                 f'finish={full.get("incomplete_reason")})')
            return full, lost_cost, None
        except API_STATUS_ERRORS as exc:
            last_error = f'attempt {attempt}: {ut.describe_exc(exc)}'
            if attempt < max_retries - 1:
                if retry_wait == 'strat1':
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                if retry_wait == 'strat2':
                    wait = ut.wait_strat2_helper(attempt)
                print(f'{label} failed: status {exc.status_code}{ut.exc_location(exc)} '
                      f'on attempt {attempt + 1}, retrying in {wait}s')
                await asyncio.sleep(wait)
                continue
            print(f'{label} failed: status {exc.status_code}, no retries left, \n '
                  f'full: {ut.report_exc(exc, f"{label}, attempt {attempt + 1}")}')
        except Exception as exc:
            last_error = f'attempt {attempt}: {ut.describe_exc(exc)}'
            if attempt < max_retries - 1:
                if retry_wait == 'strat1':
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                if retry_wait == 'strat2':
                    wait = ut.wait_strat2_helper(attempt)
                print(f'{label} failed: {ut.report_exc(exc, f"{label}, attempt {attempt + 1}")} '
                      f'on attempt {attempt + 1}, retrying in {wait}s')
                await asyncio.sleep(wait)
                continue
            print(f'{label} failed: {ut.report_exc(exc, f"{label}, attempt {attempt + 1}")}, no retries left')
    return None, lost_cost, last_error
