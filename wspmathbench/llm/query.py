"""One query function per provider, all returning the same response contract, and the local server
that serves a tiny model.

Where it sits in the flow: registry.py routes a (provider, model) pair to one of the
`*_query_async` functions here. The edit sessions call the primary LLMs through them; the scoring
code calls local_query_async, which talks to a vLLM (or plain transformers) server that this
module launches on the Linux side (cfg.shell_command).

The response contract, a dict:
    'text', 'reasoning_text', 'prompt_tokens', 'cached_tokens', 'cache_write_tokens',
    'completion_tokens', 'reasoning_tokens', 'total_tokens', 'cost_estimate', 'response',
    'status' ('completed' or 'incomplete'), 'incomplete_reason', 'server_provider'
A stream that was interrupted returns the reduced contract: 'text', 'reasoning_text', 'response'.

API keys are read from the environment when a provider is first used: OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, MOONSHOT_API_KEY and, for models reached
through OpenRouter, OPENROUTER_API_KEY. Each provider's prices are written in its function's
MODEL_CONFIG, so 'cost_estimate' is only as current as those numbers.

Retries are the caller's business (registry.retry_llm_call): every client here is built with
max_retries=0.
"""
import asyncio
import contextlib
import contextvars
import datetime
import math
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import warnings

import httpx
import requests
from anthropic import AsyncAnthropic
from openai import APIConnectionError, AsyncOpenAI

from wspmathbench import config  # not `as cfg`: below, `cfg` is a model's MODEL_CONFIG entry
from wspmathbench.llm import utils as ut

_active_timeout = contextvars.ContextVar('_active_timeout', default=None)  # the innermost deadline

@contextlib.asynccontextmanager
async def _local_timeout(seconds, label):
    """Client-side wall-clock limit on one request (None = disabled).

    Raises TimeoutError('local timeout: {label} exceeded {seconds}s') when this
    deadline fires; a TimeoutError raised inside the block by an SDK passes
    through unchanged. No partial result is returned -- retries are the caller's."""
    try:
        async with asyncio.timeout(seconds) as cm:
            token = _active_timeout.set(cm)
            try:
                yield
            finally:
                _active_timeout.reset(token)
    except TimeoutError:
        if cm.expired():  # our deadline fired, not an SDK-internal timeout
            raise TimeoutError(f'local timeout: {label} exceeded {seconds}s') from None
        raise

def _stream_result(text_parts, reasoning_parts, response):
    return {
        'text': ''.join(text_parts),
        'reasoning_text': ''.join(reasoning_parts) if reasoning_parts else None,
        'response': response,
    }

def _stream_print_delta(piece, is_reasoning, reasoning_parts, text_parts, verbose):
    # shared bookkeeping for one delta: collect, credit the live status footer, optionally echo
    ut.live_status_tick(len(piece))  # adds this delta's length to the current footer entry's count
    if is_reasoning:
        reasoning_parts.append(piece)
        if verbose: print(piece, end='', flush=True)
    else:
        if reasoning_parts and not text_parts and verbose:
            print("\n\n--- Answer ---\n", flush=True)
        text_parts.append(piece)
        if verbose: print(piece, end='', flush=True)

def _chat_delta_pieces(chunk):
    choices = getattr(chunk, 'choices', None)
    if not choices:   # usage-only terminal chunk
        return
    delta = choices[0].delta
    rc = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
    if rc: yield rc, True
    if delta.content: yield delta.content, False

GEMINI_CONTINUATION_ID = 'gemini-continuation'  # OpenRouter's id for a Gemini 4 long-decode continuation token

def _gemini_continuation_item(reasoning_details):
    """The last Gemini long-decode continuation item of an OpenRouter reasoning_details list, as
    the dict to send back unchanged (None when the list has none). A hop can carry several items
    (checkpoints); only the latest one resumes the generation."""
    found = None
    for item in reasoning_details or []:
        d = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, 'model_dump') else None)
        if d and d.get('id') == GEMINI_CONTINUATION_ID:
            found = d
    return found

async def _collect_chat_stream_async(stream, verbose=True):
    """Collect an OpenAI-chat-completions-style delta stream (async iterator). Besides the text,
    the result carries 'usage', 'finish_reason', and two OpenRouter extras that are None elsewhere:
    'native_finish_reason' and 'continuation_item' (see _gemini_continuation_item)."""
    reasoning_parts, text_parts = [], []
    usage, finish_reason, last_chunk = None, None, None
    native_finish_reason, continuation_item = None, None
    try:
        async for chunk in stream:
            last_chunk = chunk
            for piece, is_reasoning in _chat_delta_pieces(chunk):
                _stream_print_delta(piece, is_reasoning, reasoning_parts, text_parts, verbose)
            choices = getattr(chunk, 'choices', None)
            if choices and getattr(choices[0], 'finish_reason', None):
                finish_reason = choices[0].finish_reason
            if choices:
                native_finish_reason = getattr(choices[0], 'native_finish_reason', None) or native_finish_reason
                continuation_item = (_gemini_continuation_item(getattr(choices[0].delta, 'reasoning_details', None))
                                     or continuation_item)
            u = getattr(chunk, 'usage', None)
            if u is not None:
                usage = u
    except Exception as exc:
        # keep what arrived: a caller that resumes (resume_on_drop) reads it off the exception
        exc.stream_partial = _stream_result(text_parts, reasoning_parts, last_chunk)
        raise
    if verbose: print()
    result = _stream_result(text_parts, reasoning_parts, last_chunk)
    result['usage'] = usage
    result['finish_reason'] = finish_reason
    result['native_finish_reason'] = native_finish_reason
    result['continuation_item'] = continuation_item
    return result

MAX_CONTINUATION_HOPS = 20  # safety bound; 1,048,576 tokens at 65,536 per hop is 16 hops

def _openrouter_chat_hop(response, collected):
    """One OpenRouter chat-completions request as a dict, from the collected stream (stream=True)
    or the response object (collected None): 'text', 'reasoning_text', 'usage', 'finish_reason',
    'native_finish_reason', 'continuation_item', 'server_provider'. 'usage' is None for a stream
    that ended without its usage chunk (interrupted)."""
    if collected is not None:
        return {'text': collected['text'], 'reasoning_text': collected['reasoning_text'],
                'usage': collected['usage'], 'finish_reason': collected['finish_reason'],
                'native_finish_reason': collected['native_finish_reason'],
                'continuation_item': collected['continuation_item'],
                'server_provider': getattr(collected['response'], 'provider', None)}
    choice = response.choices[0]
    message = choice.message
    return {'text': message.content,
            'reasoning_text': getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None),
            'usage': response.usage, 'finish_reason': getattr(choice, 'finish_reason', None),
            'native_finish_reason': getattr(choice, 'native_finish_reason', None),
            'continuation_item': _gemini_continuation_item(getattr(message, 'reasoning_details', None)),
            'server_provider': getattr(response, 'provider', None)}

def _gemini_hop_continues(hop, spent, max_tokens):
    """Whether a Gemini long-decode hop should be followed by another request.

    Yes when it stopped on length with a continuation token and budget left: OpenRouter reports
    that as native_finish_reason 'CONTINUATION' ('MAX_TOKENS' means the max_tokens budget itself
    is spent, and resending without raising it is an error). When the native reason is missing,
    the budget is judged from `spent`, the output tokens of all hops so far."""
    if hop['finish_reason'] != 'length' or not hop['continuation_item']:
        return False
    native = hop['native_finish_reason']
    if native is not None:
        return str(native).upper() == 'CONTINUATION'
    return max_tokens is not None and spent < max_tokens

async def _collect_completion_stream_async(stream, verbose=True):
    """Collect a legacy /completions stream (continue_mode='raw'): raw text, think tags inline."""
    parts, usage, finish_reason, last_chunk = [], None, None, None
    try:
        async for chunk in stream:
            last_chunk = chunk
            if chunk.choices:
                piece = chunk.choices[0].text or ''
                if piece:
                    ut.live_status_tick(len(piece))
                    if verbose: print(piece, end='', flush=True)
                    parts.append(piece)
                finish_reason = chunk.choices[0].finish_reason or finish_reason
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
    except Exception as exc:
        exc.stream_partial = {'text': ''.join(parts), 'reasoning_text': None, 'response': last_chunk}
        raise
    if verbose: print()
    return {'text': ''.join(parts), 'reasoning_text': None, 'response': last_chunk,
            'usage': usage, 'finish_reason': finish_reason}

async def _collect_responses_stream_async(stream, verbose=True):
    """Collect an OpenAI Responses-API event stream.

    Returns (collected_text, final_response); final_response is None if the
    stream ended without a terminal event.

    Live status: both answer text and reasoning text are credited to the footer, since a
    reasoning model can stream thousands of reasoning tokens before any answer text."""
    collected_text = []
    final_response = None
    async for event in stream:
        etype = getattr(event, 'type', '')
        if etype == 'response.output_text.delta':
            delta = getattr(event, 'delta', None)
            if delta:
                ut.live_status_tick(len(delta))
                if verbose: print(delta, end='', flush=True)
                collected_text.append(delta)
        elif 'reasoning' in etype and etype.endswith('.delta'):
            # Footer credit only: the reasoning kept in the result comes from final_response.
            # OpenAI's own models send 'response.reasoning_summary_text.delta', i.e. only a
            # summary, so their footer count undercounts the reasoning tokens they generate.
            delta = getattr(event, 'delta', None)
            if delta:
                ut.live_status_tick(len(delta))
        elif etype in ('response.completed', 'response.incomplete', 'response.failed'):
            final_response = event.response
    if verbose: print()
    return ''.join(collected_text), final_response

async def _collect_genai_stream_async(stream, verbose=True):
    """Collect a google-genai generate_content stream (thought-flagged parts).

    Also returns the raw 'chunks', from which google_query_async reads finish_reason,
    prompt_feedback, and usage_metadata to build the full contract."""
    reasoning_parts, text_parts, chunks = [], [], []
    async for chunk in stream:
        chunks.append(chunk)
        for cand in (chunk.candidates or []):
            parts = (cand.content.parts if (cand.content and cand.content.parts) else [])
            for part in parts:
                piece = getattr(part, 'text', None)
                if not piece: continue
                _stream_print_delta(piece, getattr(part, 'thought', False),
                                    reasoning_parts, text_parts, verbose)
    if verbose: print()
    result = _stream_result(text_parts, reasoning_parts, chunks[-1] if chunks else None)
    result['chunks'] = chunks
    return result

_async_clients = {}

def _get_async_client(api_key_env, base_url=None):
    key = (api_key_env, base_url)
    if key not in _async_clients:
        _async_clients[key] = AsyncOpenAI(
            api_key=os.environ[api_key_env], base_url=base_url, max_retries=0,
            timeout=5000,
        )
    return _async_clients[key]

_genai_clients = {}

def _get_genai_client(api_key_env):
    from google import genai
    key = (api_key_env)
    if key not in _genai_clients:
        _genai_clients[key] = genai.Client(
            api_key=os.environ[api_key_env])
    return _genai_clients[key]

_anthropic_clients = {}

_stale_anthropic_clients = []

def _get_anthropic_client(api_key_env):
    # One client per event loop: a connection pool left over from a closed asyncio.run()
    # loop fails with 'Event loop is closed', so a new loop gets a new client. Superseded
    # clients are kept referenced, because their finalizer would schedule a close against
    # the dead loop and log 'Task exception was never retrieved'.
    # max_retries=0: retries are the caller's business (registry.retry_llm_call).
    loop = asyncio.get_running_loop()
    cached = _anthropic_clients.get(api_key_env)
    if cached is None or cached[0] is not loop:
        if cached is not None: _stale_anthropic_clients.append(cached)
        _anthropic_clients[api_key_env] = (loop, AsyncAnthropic(
            api_key=os.environ[api_key_env], max_retries=0, timeout=5000,
        ))
    return _anthropic_clients[api_key_env][1]

_local_clients = {}

LOCAL_REQUEST_TIMEOUT = 3600.0  # seconds; scale with max_tokens / per-stream tok/s

_openrouter_http_clients = {}

_stale_openrouter_http_clients = []

def _get_openrouter_http_client():
    # One shared client per event loop, as in _get_anthropic_client. A fresh client per lookup
    # would open a new TLS connection each time, and with many lookups in flight those time out.
    loop = asyncio.get_running_loop()
    cached = _openrouter_http_clients.get('generation')
    if cached is None or cached[0] is not loop:
        if cached is not None: _stale_openrouter_http_clients.append(cached)
        _openrouter_http_clients['generation'] = (loop, httpx.AsyncClient(timeout=10))
    return _openrouter_http_clients['generation'][1]

async def _openrouter_provider_lookup(gen_id, api_key, retries=5, delay=1.0):
    """Best-effort lookup of the upstream provider that served an OpenRouter generation.

    Runs after the answer has already been generated and billed, and only fills in the
    server_provider metadata, so a network failure returns None instead of raising: a
    lookup error must never cost a paid answer (or trigger a retry of it)."""
    http = _get_openrouter_http_client()
    try:
        for attempt in range(retries):
            r = await http.get("https://openrouter.ai/api/v1/generation",
                               params={"id": gen_id},
                               headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 200:
                return (r.json().get("data") or {}).get("provider_name")
            await asyncio.sleep(delay * (attempt + 1))   # 404 until the record lands
    except (httpx.HTTPError, TimeoutError):
        pass
    return None

async def openai_query_async(prompt, model='gpt-5-nano', reasoning_effort='medium', tool=None,
                             system_prompt=None, max_tokens=None, temperature=None, top_p=None,
                             response_format=None, stream=False, answer_verbosity=None, flex=True,
                             verbose=True,  prompt_cache_key=None, prompt_cache_retention=None,
                             prompt_cache_options=None, host=None, local_timeout=None):
    """Query an OpenAI model through the Responses API.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        A key of MODEL_CONFIG below.
    reasoning_effort : str
        Reasoning effort; the valid values depend on the model.
    tool : str or None
        Optional server-side tool ('web_search').
    system_prompt : str or None
        System-level instructions. They are part of the cached prefix, so they must be identical
        across requests that should share a cache.
    max_tokens : int or None
        Maximum number of output tokens (the API's max_output_tokens).
    temperature, top_p : float or None
        Sampling parameters. Not supported by every model.
    response_format : dict or None
        Response format specification, e.g. {"type": "json_object"}.
    stream : bool
        Stream the response. Usage is still returned, read from the terminal stream event.
    answer_verbosity : str or None
        The API's text verbosity, for the models that take it.
    flex : bool
        Use the flex service tier (half price, slower) where the model supports it.
    verbose : bool
        Print the response text, usage and cost estimate.
    prompt_cache_key : str or None
        Routing hint so that requests sharing a long prefix land on the same cache machine.
        OpenAI platform only.
    prompt_cache_retention : str or None
        '24h', 'in_memory' or None (account default), for the models that take it. Dropped for the
        models that use prompt_cache_options instead.
    prompt_cache_options : dict or None
        For the models that take it, passed through as given, e.g. {'mode': 'implicit'}.
    host : None or 'OpenRouter'
        None -> OpenAI directly.
    local_timeout : float or None
        Client-side wall-clock limit in seconds on the request, stream drain included
        (None = disabled). Raises TimeoutError; see _local_timeout.

    Returns
    -------
    dict
        The response contract described at the top of this module.
    """

    # Prices in dollars per token. A cache write is billed at 1.25x the input price and a cache
    # read at 0.1x; flex is half price throughout.
    MODEL_CONFIG = {
        'gpt-6-astra': {
            'supports_reasoning': True,
            'reasoning_options': ['low', 'medium', 'high', 'xhigh', 'max'],
            'supports_temperature': False,
            'supports_flex': True,
            'uses_cache_options': True,
            'cost_in': 10 / 1e6,
            'cost_cached': 1 / 1e6,
            'cost_cache_write': 12.5 / 1e6,
            'cost_out': 50 / 1e6,
            'cost_in_flex': 5 / 1e6,
            'cost_cached_flex': 0.5 / 1e6,
            'cost_cache_write_flex': 6.25 / 1e6,
            'cost_out_flex': 25 / 1e6,
        },
        # Served as gpt-6-astra with the reasoning mode 'pro'.
        'gpt-6-astra-pro': {
            'supports_reasoning': True,
            'reasoning_options': ['low', 'medium', 'high', 'xhigh', 'max'],
            'supports_temperature': False,
            'supports_flex': True,
            'uses_cache_options': True,
            'cost_in': 10 / 1e6,
            'cost_cached': 1 / 1e6,
            'cost_cache_write': 12.5 / 1e6,
            'cost_out': 50 / 1e6,
            'cost_in_flex': 5 / 1e6,
            'cost_cached_flex': 0.5 / 1e6,
            'cost_cache_write_flex': 6.25 / 1e6,
            'cost_out_flex': 25 / 1e6,
        },
        'gpt-6.1-sol': {
            'supports_reasoning': True,
            'reasoning_options': ['low', 'medium', 'high', 'xhigh', 'max'],
            'supports_temperature': False,
            'supports_flex': True,
            'uses_cache_options': True,
            'cost_in': 2 / 1e6,
            'cost_cached': 0.10 / 1e6,
            'cost_cache_write': 2.50 / 1e6,
            'cost_out': 10 / 1e6,
            'cost_in_flex': 1 / 1e6,
            'cost_cached_flex': 0.05 / 1e6,
            'cost_cache_write_flex': 1.25 / 1e6,
            'cost_out_flex': 5 / 1e6,
        },
        # The small model the scoring code uses to read a final answer out of a reply.
        'gpt-5.6-luna': {
            'supports_reasoning': True,
            'reasoning_options': ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
            'supports_temperature': False,
            'supports_flex': True,
            'uses_cache_options': True,
            'cost_in': 1 / 1e6,
            'cost_cached': 0.1 / 1e6,
            'cost_cache_write': 1.25 / 1e6,
            'cost_out': 6 / 1e6,
            'cost_in_flex': 0.1 / 1e6,
            'cost_cached_flex': 0.01 / 1e6,
            'cost_cache_write_flex': 0.125 / 1e6,
            'cost_out_flex': 0.6 / 1e6,
        },
        'gpt-5-nano': {
            'supports_reasoning': True,
            'reasoning_options': [None, 'minimal', 'low', 'medium', 'high'],
            'supports_temperature': False,
            'supports_flex': True,
            'supports_extended_cache': False,  # in-memory caching only
            'cost_in': 0.05 / 1e6,
            'cost_cached': 0.005 / 1e6,
            'cost_out': 0.4 / 1e6,
            'cost_in_flex': 0.025 / 1e6,
            'cost_cached_flex': 0.00025 / 1e6,
            'cost_out_flex': 0.2 / 1e6,
        },
    }

    TOOL_OPTIONS = ['web_search']
    RETENTION_OPTIONS = ['in_memory', '24h']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Validate host ---
    host_options = [None, 'OpenRouter']  # None = OpenAI directly
    if host not in host_options:
        raise ValueError(f"host `{host}` not in allowed hosts for '{model}': `{host_options}`.")

    # --- Validate reasoning effort ---
    if cfg['supports_reasoning'] == True:
        if reasoning_effort not in cfg['reasoning_options']:
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. "
                f"Must be one of {cfg['reasoning_options']}"
            )
    if cfg['supports_reasoning'] == False:
        if reasoning_effort != None:
            if verbose==True:
                print(f"'{model}' does not support reasoning. Setting reasoning_effort to 'None'")
            reasoning_effort = None

    # --- Validate verbosity options ---
    if answer_verbosity != None:
        if answer_verbosity not in cfg['verbosity_options']:
            if verbose == True:
                print(f"Invalid answer_verbosity '{answer_verbosity}' for model '{model}'. "
                      f"Must be one of {cfg['verbosity_options']} \n"
                      f"Setting answer_verbosity to 'medium'")
            answer_verbosity='medium'
    # --- Validate tool ---
    if tool is not None and tool not in TOOL_OPTIONS:
        raise ValueError(f"Invalid tool '{tool}'. Must be one of {TOOL_OPTIONS}")

    # --- Validate temperature support ---
    if temperature is not None and not cfg['supports_temperature']:
        print(f"Model '{model}' does not support the temperature parameter. Setting temperature=None")
        temperature = None

    if flex == True and not cfg['supports_flex']:
        if verbose == True:
            print(f"Model '{model}' does not support the flex_process. Disabling flex.")
        flex = False

    # --- Caching parameters: a model takes prompt_cache_options or prompt_cache_retention ---
    if prompt_cache_retention is not None and cfg.get('uses_cache_options', False):
        print(f"Model '{model}' does not take prompt_cache_retention: its implicit caching is "
              f"automatic. Use prompt_cache_options for explicit control. Dropping "
              f"prompt_cache_retention.")
        prompt_cache_retention = None
    if prompt_cache_options is not None and not cfg.get('uses_cache_options', False):
        print(f"Model '{model}' does not take prompt_cache_options. Dropping it; "
              f"use prompt_cache_retention instead.")
        prompt_cache_options = None

    # --- Validate prompt_cache_retention ---
    if prompt_cache_retention is not None:
        if prompt_cache_retention not in RETENTION_OPTIONS:
            raise ValueError(
                f"Invalid prompt_cache_retention '{prompt_cache_retention}'. "
                f"Must be one of {RETENTION_OPTIONS}"
            )
        if prompt_cache_retention == '24h' and not cfg.get('supports_extended_cache', False):
            print(f"Model '{model}' does not support extended (24h) prompt cache retention. "
                  f"Falling back to the API default (in-memory, ~5-10 min).")
            prompt_cache_retention = None
        if prompt_cache_retention == 'in_memory' and not cfg.get('supports_in_memory_cache', True):
            print(f"Model '{model}' only supports '24h' retention. Falling back to the API default.")
            prompt_cache_retention = None

    # --- Build API kwargs ---
    extra_body = {}
    if host == 'OpenRouter':
        extra_body['provider'] = {
            'require_parameters': True,
            'sort': 'throughput',
            'ignore': ['Together', 'SiliconFlow']
        }
        client = _get_async_client("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
    else:  # OpenAI directly
        client = _get_async_client("OPENAI_API_KEY")

    model_ = f"openai/{model}" if host == 'OpenRouter' else model
    if host != 'OpenRouter':
        model_ = model_.replace('-pro', '')
    kwargs = {
        'model': model_,
        'input': prompt,
        'stream': stream,
    }
    reasoning_mode = 'pro' if 'pro' in model else 'standard'
    if cfg['supports_reasoning']: kwargs['reasoning'] = {
        'effort': reasoning_effort, 'summary': 'auto', 'mode': reasoning_mode
    }
    if answer_verbosity: kwargs['text'] = {"format": {"type": "text"}, 'verbosity': answer_verbosity}
    if system_prompt is not None: kwargs['instructions'] = system_prompt
    if max_tokens is not None: kwargs['max_output_tokens'] = max_tokens
    if temperature is not None: kwargs['temperature'] = temperature
    if top_p is not None: kwargs['top_p'] = top_p
    if response_format is not None: kwargs['text'] = {'format': response_format}
    if flex: kwargs['service_tier'] = 'flex'
    if tool == 'web_search':
        kwargs['tools'] = [{'type': 'web_search'}]
        kwargs['include'] = ['web_search_call.action.sources']

    # --- Caching parameters (OpenAI platform only) ---
    if host == 'OpenRouter':
        if extra_body: kwargs['extra_body'] = extra_body
        # prompt_cache_key/_retention/_options are OpenAI-platform parameters and are not sent
        # through OpenRouter. (OpenAI's implicit caching still applies upstream there and is
        # reported in cached_tokens.)
    else:
        if prompt_cache_key is not None: kwargs['prompt_cache_key'] = prompt_cache_key
        if prompt_cache_retention is not None: kwargs['prompt_cache_retention'] = prompt_cache_retention
        if prompt_cache_options is not None: kwargs['prompt_cache_options'] = prompt_cache_options

    # --- Call the API ---
    async with _local_timeout(local_timeout, f'{model} via {host or "OpenAI"}'):
        response = await client.responses.create(**kwargs)
        if stream:
            collected_text, final_response = await _collect_responses_stream_async(response, verbose=verbose)

    if verbose:
        print(f'full response: {response}\n\n')

    if host == 'OpenRouter':
        # --- Extract content ---
        if stream:
            if final_response is None:
                # Stream ended without a terminal event; return what we have.
                return _stream_result([collected_text], [], response)
            response = final_response
            text = collected_text or response.output_text
        else:
            text = response.output_text

        # --- Extract usage and compute cost ---
        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = usage.total_tokens

        # Cache hits as the serving provider reports them.
        details = (getattr(usage, 'input_tokens_details', None)
                   or getattr(usage, 'prompt_tokens_details', None))
        cached_tokens = getattr(details, 'cached_tokens', 0) or 0
        cache_write_tokens = getattr(details, 'cache_write_tokens', 0) or 0
        cdetails = (getattr(usage, 'output_tokens_details', None)
                    or getattr(usage, 'completion_tokens_details', None))
        reasoning_tokens = (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0
        server_provider = getattr(response, 'provider', None)
        if server_provider is None:
            server_provider = await _openrouter_provider_lookup(
                response.id, os.environ["OPENROUTER_API_KEY"])

        cost = getattr(usage, 'cost', None)
        if cost is None: cost = 0.0

        # Responses-API objects may not expose .choices
        choices = getattr(response, 'choices', None)
        finish_reason = getattr(choices[0], 'finish_reason', None) if choices else None
        status = 'incomplete' if finish_reason == 'length' else 'completed'
        incomplete_reason = 'max_output_tokens' if finish_reason == 'length' else None

    else:
        # --- Handle streaming vs. non-streaming ---
        if stream:
            if final_response is None:
                # Stream ended without a terminal event; return what we have.
                return _stream_result([collected_text], [], response)
            response = final_response
            text = collected_text or response.output_text
        else:
            text = response.output_text

        # --- Extract usage and compute cost (shared by stream & non-stream) ---
        prompt_tokens = response.usage.input_tokens
        details = response.usage.input_tokens_details
        cached_tokens = details.cached_tokens
        # Cache writes are reported, and billed at 1.25x input, by the models that have
        # 'cost_cache_write'; absent or zero on the others.
        cache_write_tokens = getattr(details, 'cache_write_tokens', 0) or 0
        completion_tokens = response.usage.output_tokens
        odetails = getattr(response.usage, 'output_tokens_details', None)
        reasoning_tokens = (getattr(odetails, 'reasoning_tokens', 0) or 0) if odetails else 0
        total_tokens = response.usage.total_tokens
        server_provider = 'OpenAI'

        status = response.status
        incomplete_reason = getattr(response.incomplete_details, 'reason', None) if status == 'incomplete' else None

        non_cached_input = prompt_tokens - cached_tokens - cache_write_tokens
        if flex == False:
            cost = (
                    non_cached_input * cfg['cost_in']
                    + cached_tokens * cfg['cost_cached']
                    + cache_write_tokens * cfg.get('cost_cache_write', cfg['cost_in'])
                    + completion_tokens * cfg['cost_out']
            )
        else:
            cost = (
                    non_cached_input * cfg['cost_in_flex']
                    + cached_tokens * cfg['cost_cached_flex']
                    + cache_write_tokens * cfg.get('cost_cache_write_flex', cfg['cost_in_flex'])
                    + completion_tokens * cfg['cost_out_flex']
            )

    reasoning = '\n'.join(
        s.text
        for item in response.output
        if getattr(item, 'type', None) == 'reasoning'
        for s in (getattr(item, 'summary', None) or [])
    ) or None

    if verbose:
        if reasoning:
            print(f'.--- Reasoning ---.\n {reasoning} \n\n')
        print(f'.--- Response ---.\n {text}')
        print(f"\nTokens — input: {prompt_tokens} (cached: {cached_tokens}, "
              f"cache writes: {cache_write_tokens}), "
              f"output: {completion_tokens}, reasoning: {reasoning_tokens}, "
              f"total: {total_tokens}, server provider: {server_provider}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': text,
        'reasoning_text': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'cache_write_tokens': cache_write_tokens,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
    }

async def deepseek_query_async(prompt, model='deepseek-v4.1-flash-1017', system_prompt=None, thinking=False,
                   reasoning_effort=None,
                   max_tokens=None, temperature=None, top_p=None, frequency_penalty=None,
                   presence_penalty=None, response_format=None, tools=None, stream=False,
                   verbose=True, host=None, local_timeout=None):
    """Query a DeepSeek model through an OpenAI-compatible chat completions endpoint: DeepSeek's
    own API, or OpenRouter.

    Terminology, here and in the other query functions: the *provider* is the maker of a model,
    the *host* is the platform the API call goes through (DeepSeek or OpenRouter), and the *server
    provider* is whoever actually runs the model that answered (reported by OpenRouter).

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        A key of MODEL_CONFIG below.
    thinking : bool
        Thinking mode, for the models in EFFORT_MODELS below: sent as
        extra_body={'thinking': {'type': 'enabled' or 'disabled'}} on DeepSeek's API and as
        reasoning.enabled on OpenRouter. In thinking mode the API ignores temperature, top_p and
        the penalties. For any other model nothing is sent and the server's default applies.
    reasoning_effort : {'low', 'medium', 'high', 'max'} or None
        Depth of the reasoning, for the models in EFFORT_MODELS; it overrides thinking=False.
        For any other model it is not sent (a note is printed when verbose).
    system_prompt : str or None
        Sent as the system message when given.
    max_tokens : int or None
        Maximum number of output tokens, reasoning included.
    temperature, top_p, frequency_penalty, presence_penalty : float or None
        Sampling parameters, sent only when given.
    response_format : dict or None
        Response format specification, e.g. {"type": "json_object"}.
    tools : list or None
        Tool definitions for function calling, passed through.
    stream : bool
        Stream the response.
    verbose : bool
        Print the response text, the reasoning, usage and the cost estimate.
    host : None or 'OpenRouter'
        None -> DeepSeek's own API (DEEPSEEK_API_KEY); 'OpenRouter' -> OPENROUTER_API_KEY.
    local_timeout : float or None
        Client-side wall-clock limit in seconds on the request, stream drain included
        (None = disabled). Raises TimeoutError; see _local_timeout.

    Returns
    -------
    dict
        The response contract described at the top of this module. OpenRouter's billed cost is
        used as 'cost_estimate' when it is reported.
    """
    # Prices in dollars per token.
    MODEL_CONFIG = {
        'deepseek-v4-flash': {
            'supports_temperature': True,  # only in non-thinking mode
            'context_length': 1_000_000,
            'max_output_default': 8_000,
            'max_output': 384_000,
            'cost_in_hit': 0.0028 / 1e6,
            'cost_in_miss': 0.14 / 1e6,
            'cost_out': 0.28 / 1e6,
        },
        'deepseek-v4.1-flash-1017': {
            'supports_temperature': True,  # only in non-thinking mode
            'context_length': 1_000_000,
            'max_output_default': 200_000,
            'max_output': 384_000,
            'cost_in_hit': 0.0028 / 1e6,
            'cost_in_miss': 0.14 / 1e6,
            'cost_out': 0.28 / 1e6,
        },
    }
    # The models for which `thinking` and `reasoning_effort` are sent.
    EFFORT_MODELS = ('deepseek-v4-flash', 'deepseek-v4.1-flash-1017')

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    if reasoning_effort != None and thinking == False:
        print(f'Conflicting reasoning_effort, {reasoning_effort}, and thinking, {thinking}. '
              f'Overriding thinking to True')
    if reasoning_effort == None and thinking != False:
        print(f'Conflicting reasoning_effort, {reasoning_effort}, and thinking, {thinking}. '
              f'Overriding thinking to False')

    if reasoning_effort is not None and model not in EFFORT_MODELS:
        if verbose == True:
            print(f"reasoning_effort is only sent for {EFFORT_MODELS}; will be ignored.")

    # --- Warn about temperature in thinking mode ---
    if thinking and temperature is not None:
        warnings.warn(
            "temperature has no effect in thinking mode (silently ignored by the API).",
            stacklevel=2,
        )

    # --- Build messages ---
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream,
        'timeout': 3600,
    }

    extra_body = {}
    # --- Select endpoint and credentials ---
    if host == 'OpenRouter':
        api_key = os.environ["OPENROUTER_API_KEY"]
        base_url = "https://openrouter.ai/api/v1"
        extra_body['provider'] = {
            'require_parameters': True,
            'sort': 'throughput',
            'ignore': ['Cloudflare']}
    else:
        api_key = os.environ["DEEPSEEK_API_KEY"]
        base_url = "https://api.deepseek.com"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # --- Thinking-mode control via extra_body ---
    # The API thinks by default; the mode is sent explicitly so that the function argument decides.
    if reasoning_effort != None:
        thinking = True
    if model in EFFORT_MODELS:
        if host == 'OpenRouter':
            extra_body['reasoning'] = {'enabled': bool(thinking)}
            if thinking == True and reasoning_effort != None:
                extra_body['reasoning']['effort'] = reasoning_effort
        else:
            extra_body['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
            if reasoning_effort != None and thinking == True:
                kwargs['reasoning_effort'] = reasoning_effort
    if extra_body: kwargs['extra_body'] = extra_body
    if max_tokens is not None: kwargs['max_tokens'] = max_tokens
    if temperature is not None: kwargs['temperature'] = temperature
    if top_p is not None: kwargs['top_p'] = top_p
    if frequency_penalty is not None: kwargs['frequency_penalty'] = frequency_penalty
    if presence_penalty is not None: kwargs['presence_penalty'] = presence_penalty
    if response_format is not None: kwargs['response_format'] = response_format
    if tools is not None: kwargs['tools'] = tools

    # --- Call the API ---
    async with _local_timeout(local_timeout, f'{model} via {host or "DeepSeek"}'):
        response = await client.chat.completions.create(**kwargs)
        if stream:
            collected = await _collect_chat_stream_async(response, verbose=verbose)
    if verbose:
        print(f'response full: {response}')
    # --- Handle streaming ---
    if stream:
        if collected['usage'] is None:
            return collected  # interrupted stream: degrade to reduced contract
        output_text = collected['text']
        reasoning = collected['reasoning_text']
        usage = collected['usage']
        finish_reason = collected['finish_reason']
        server_provider = getattr(collected['response'], 'provider', None)
    # --- Extract content ---
    else:
        message = response.choices[0].message
        output_text = message.content
        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
        usage = response.usage
        finish_reason = getattr(response.choices[0], 'finish_reason', None)
        server_provider = getattr(response, 'provider', None)

    # --- Extract usage and compute cost ---
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    cdetails = getattr(usage, 'completion_tokens_details', None)
    reasoning_tokens = (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0

    status = 'incomplete' if finish_reason == 'length' else 'completed'
    incomplete_reason = 'max_tokens' if finish_reason == 'length' else None

    cached_tokens = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
    if cached_tokens:
        print(f'cache hit of {cached_tokens}')
    cache_miss_tokens = getattr(usage, 'prompt_cache_miss_tokens', 0) or 0

    cost_estimate = (
        cached_tokens * cfg['cost_in_hit']
        + cache_miss_tokens * cfg['cost_in_miss']
        + completion_tokens * cfg['cost_out']
    )
    cost_actual = getattr(usage, 'cost', None)

    cost = cost_estimate if cost_actual is None else cost_actual

    if verbose:
        print(".--- Reasoning ---.")
        print(reasoning)
        print("\n.--- Answer ---.")
        print(output_text)
        print(f"\nTokens -- prompt: {prompt_tokens} (cache hit: {cached_tokens}, "
              f"cache miss: {cache_miss_tokens}), completion: {completion_tokens}, "
              f"total: {total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'reasoning_text': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'cache_write_tokens': 0,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
    }

async def kimi_query_async(prompt, model='kimi-k3.1-pro', system_prompt=None, thinking=None,
                           reasoning_effort=None, max_tokens=None, temperature=None, top_p=None,
                           response_format=None, tools=None, tool_choice=None, stream=False,
                           verbose=True, host=None):
    """Query a Moonshot AI Kimi model through Moonshot's own API or OpenRouter.

    Both hosts are reached through OpenAI-compatible chat completions. With stream=False a single
    non-streamed request is made; with stream=True the response is streamed (usage chunk
    requested) and drained by _collect_chat_stream_async.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        A key of MODEL_CONFIG below.
    system_prompt : str or None
        Sent as the system message when given.
    thinking : bool or None
        What it means depends on the model's 'thinking_control': on a 'toggle' model True/False
        switch thinking (None -> True); an 'effort' model (kimi-k3.1-pro) and an 'always' model
        always think, so None or True is accepted and False raises; on a 'never' model True raises.
    reasoning_effort : str or None
        For an 'effort' model: one of its 'effort_options'. None sends the model's default
        explicitly, which is the same as omitting it. Sent as the top-level reasoning_effort on
        Moonshot and as reasoning.effort on OpenRouter. Raises on a model without effort control.
    max_tokens : int or None
        Output cap, reasoning included. None -> the model's 'max_tokens_default'. Checked against
        its 'max_output'.
    temperature, top_p : float or None
        Sampling parameters, sent only when given. Dropped (with a note when verbose) on a
        'fixed_sampling' model: OpenRouter lists neither as supported for kimi-k3.1-pro, so with
        require_parameters a value would exclude every provider.
    response_format : dict or None
        e.g. {"type": "json_object"}. Passed through on both hosts.
    tools : list or None
        Function definitions, passed through. Moonshot's builtin functions are available on
        Moonshot's own API only and raise with host='OpenRouter'.
    tool_choice : str, dict, or None
        Passed through.
    stream : bool
        Stream the response (with verbose, reasoning and answer deltas are echoed live).
    verbose : bool
        Print the full response object, the reasoning and answer (non-streamed path), usage and
        the cost estimate.
    host : None or 'OpenRouter'
        None -> Moonshot's own API (api.moonshot.ai/v1, MOONSHOT_API_KEY). 'OpenRouter' -> chat
        completions through OpenRouter (OPENROUTER_API_KEY) with require_parameters; only for a
        model with an 'openrouter_name'.

    Returns
    -------
    dict
        The response contract described at the top of this module; 'response' is the completion
        object, or the final chunk when streamed. A stream that closes with no finish_reason and
        no usage raises StreamInterruptedError (retryable); one that finished but sent no usage
        returns the reduced contract.

    Notes
    -----
    prompt_tokens includes cache reads (and writes, where reported), so cached_tokens is a subset
    of it, as in the other query functions. completion_tokens includes reasoning on both hosts.
    reasoning_tokens is what the host reports (completion_tokens_details), 0 where it reports
    nothing, so it is not comparable between hosts.

    A content-filter stop (finish_reason='content_filter') comes back as an ordinary failed
    attempt: text='' (not None, so retry_llm_call does not retry it) and reasoning_text=None, as
    in anthropic_query_async.

    Cache writes are costed at the cache-miss rate where no separate price is recorded.
    OpenRouter's billed usage.cost is used as cost_estimate when present.
    """

    # Prices in dollars per token.
    MODEL_CONFIG = {
        # Always thinks; the depth is set by reasoning_effort (default max). No temperature or
        # top_p on OpenRouter. The output cap is that of OpenRouter's top provider.
        'kimi-k3.1-pro': {
            'openrouter_name': 'moonshotai/kimi-k3.1-pro',
            'context_length': 1_048_576,
            'max_output': 943_718,
            'max_tokens_default': 200000,
            'thinking_control': 'effort',
            'effort_options': ['low', 'high', 'max'],
            'default_effort': 'max',
            'fixed_sampling': True,
            'prices': {
                'Moonshot': {'in_miss': 2.00 / 1e6, 'in_hit': 0.20 / 1e6,
                             'cache_write': None, 'out': 10.00 / 1e6},
                'OpenRouter': {'in_miss': 2.00 / 1e6, 'in_hit': 0.20 / 1e6,
                               'cache_write': None, 'out': 10.00 / 1e6},
            },
        },
    }

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Resolve host ---
    allowed_hosts = [None, 'OpenRouter']
    if host not in allowed_hosts:
        raise ValueError(f'host `{host}` not in allowed hosts: `{allowed_hosts}`.')
    if host == 'OpenRouter' and not cfg['openrouter_name']:
        raise ValueError(f"Model '{model}' is not wired for OpenRouter; use host=None.")
    if host == 'OpenRouter' and tools is not None and any(
            isinstance(t, dict) and t.get('type') == 'builtin_function' for t in tools):
        raise ValueError("Moonshot builtin tools are only available on Moonshot's own API "
                         "(host=None).")
    host_label = host or 'Moonshot'
    label = f'{model} via {host_label}'

    # --- Validate thinking / reasoning effort ---
    ctl = cfg['thinking_control']
    if thinking is not None and not isinstance(thinking, bool):
        raise ValueError(f"thinking must be a bool or None, got {thinking!r}.")
    if ctl == 'effort':
        if thinking is False:
            raise ValueError(f"Model '{model}' always thinks (no non-thinking mode on Moonshot's API). "
                             f"Use a lower reasoning_effort instead.")
        if reasoning_effort is None:
            reasoning_effort = cfg['default_effort']  # sent explicitly; same as omitting it
        if reasoning_effort not in cfg['effort_options']:
            raise ValueError(f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. "
                             f"Must be one of {cfg['effort_options']}")
        thinking = True
    else:
        if reasoning_effort is not None:
            raise ValueError(f"Model '{model}' does not support reasoning_effort.")
        if ctl == 'toggle':
            thinking = True if thinking is None else thinking
        elif ctl == 'always':
            if thinking is False:
                raise ValueError(f"Model '{model}' always thinks; thinking=False is not available.")
            thinking = True
        else:  # 'never'
            if thinking:
                raise ValueError(f"Model '{model}' has no thinking mode.")
            thinking = False

    # --- Validate sampling parameters ---
    if cfg['fixed_sampling']:
        dropped = [name for name, value in (('temperature', temperature), ('top_p', top_p))
                   if value is not None]
        if dropped and verbose:
            print(f"Model '{model}' does not take sampling parameters. Dropping {dropped}.")
        temperature = top_p = None

    # --- Validate max tokens ---
    if max_tokens is None:
        max_tokens = cfg['max_tokens_default']
    if max_tokens is not None and cfg['max_output'] is not None and max_tokens > cfg['max_output']:
        raise ValueError(f"max_tokens={max_tokens} exceeds the {cfg['max_output']} output cap of '{model}'.")

    # --- Build messages ---
    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # --- Build API kwargs ---
    # extra_body is built ONCE and only ever added to (merge, never reassign).
    extra_body = {}
    kwargs = {
        'messages': messages,
        'stream': stream,
    }
    if stream:
        kwargs['stream_options'] = {'include_usage': True}  # so the usage chunk arrives
    if host == 'OpenRouter':
        client = _get_async_client("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
        kwargs['model'] = cfg['openrouter_name']
        extra_body['provider'] = {'require_parameters': True}
        if ctl == 'effort':
            extra_body['reasoning'] = {'enabled': True, 'effort': reasoning_effort}
        elif ctl == 'toggle':
            extra_body['reasoning'] = {'enabled': thinking}
    else:  # Moonshot's own API
        client = _get_async_client("MOONSHOT_API_KEY", "https://api.moonshot.ai/v1")
        kwargs['model'] = model
        if ctl == 'effort':
            kwargs['reasoning_effort'] = reasoning_effort  # a top-level field
        elif ctl == 'toggle':
            extra_body['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
        # 'always' / 'never' models: the weights decide; nothing is sent
    if extra_body: kwargs['extra_body'] = extra_body
    if max_tokens is not None: kwargs['max_tokens'] = max_tokens
    if temperature is not None: kwargs['temperature'] = temperature
    if top_p is not None: kwargs['top_p'] = top_p
    if response_format is not None: kwargs['response_format'] = response_format
    if tools is not None: kwargs['tools'] = tools
    if tool_choice is not None: kwargs['tool_choice'] = tool_choice

    # --- Call the API ---
    response = await client.chat.completions.create(**kwargs)
    if stream:
        collected = await _collect_chat_stream_async(response, verbose=verbose)
        response = collected['response']  # the final chunk (carries id, provider, usage)
    if verbose:
        print(f'response full: {response}')

    # --- Extract content ---
    if stream:
        finish_reason = collected['finish_reason']
        usage = collected['usage']
        # Some Moonshot models attach usage to the final choice rather than to the chunk; read
        # it there when the top-level usage chunk is absent.
        if usage is None and getattr(response, 'choices', None):
            raw_usage = getattr(response.choices[0], 'usage', None)
            if isinstance(raw_usage, dict):
                from openai.types import CompletionUsage
                raw_usage = CompletionUsage.model_validate(raw_usage)
            usage = raw_usage
        if usage is None:
            if finish_reason is None:
                n_chars = len(collected['text'] or '') + len(collected['reasoning_text'] or '')
                raise StreamInterruptedError(
                    f'{label}: stream closed ' + ('without sending any chunk' if n_chars == 0 else
                                                  f'unfinished after {n_chars} chars'))
            return collected  # finished but no usage chunk: degrade to reduced contract
        text = collected['text'] or ''
        reasoning = collected['reasoning_text']
    else:
        choice = response.choices[0]
        message = choice.message
        text = message.content or ''
        reasoning = (getattr(message, 'reasoning_content', None)  # Moonshot
                     or getattr(message, 'reasoning', None))      # OpenRouter
        finish_reason = getattr(choice, 'finish_reason', None)
        usage = response.usage
    refused = (finish_reason == 'content_filter')

    # --- Extract usage and compute cost ---
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens  # includes reasoning
    total_tokens = usage.total_tokens
    pdetails = getattr(usage, 'prompt_tokens_details', None)
    cached_tokens = (getattr(usage, 'prompt_cache_hit_tokens', None)        # DeepSeek-style
                     or getattr(usage, 'cached_tokens', None)               # Moonshot-style
                     or ((getattr(pdetails, 'cached_tokens', 0) or 0) if pdetails else 0))
    cache_write_tokens = (getattr(pdetails, 'cache_write_tokens', 0) or 0) if pdetails else 0
    cdetails = getattr(usage, 'completion_tokens_details', None)
    reasoning_tokens = (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0
    uncached_input = prompt_tokens - cached_tokens - cache_write_tokens

    price = cfg['prices'][host_label]
    cost_est = (
            uncached_input * price['in_miss']
            + cached_tokens * price['in_hit']
            + cache_write_tokens * (price['cache_write'] if price['cache_write'] is not None
                                    else price['in_miss'])
            + completion_tokens * price['out']
    )
    cost_actual = getattr(usage, 'cost', None) if host == 'OpenRouter' else None
    cost = cost_est if cost_actual is None else cost_actual

    if host == 'OpenRouter':
        server_provider = getattr(response, 'provider', None)
        if server_provider is None and getattr(response, 'id', None):
            server_provider = await _openrouter_provider_lookup(
                response.id, os.environ["OPENROUTER_API_KEY"])
    else:
        server_provider = 'Moonshot'

    # --- Shared post-processing ---
    if refused:
        # Scored like any other no-attempt failure: empty text (not None, so retry_llm_call does
        # not retry it) and nothing that marks it as a refusal.
        text, reasoning = '', None

    truncated = (finish_reason == 'length')
    status = 'incomplete' if truncated else 'completed'
    incomplete_reason = 'max_tokens' if truncated else None

    if verbose:
        if not stream:  # streamed output was already echoed
            if reasoning:
                print(f'.--- Reasoning ---.\n {reasoning} \n\n')
            print(f'.--- Response ---.\n {text}')
        print(f"\nTokens — input: {prompt_tokens} (cached: {cached_tokens}, "
              f"cache writes: {cache_write_tokens}), "
              f"output: {completion_tokens} (reasoning: {reasoning_tokens}), "
              f"total: {total_tokens}, server provider: {server_provider}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': text,
        'reasoning_text': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'cache_write_tokens': cache_write_tokens,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
    }

async def anthropic_query_async(prompt, model='claude-fable-5-2', system_prompt=None,
                                thinking=None, thinking_budget=None, reasoning_effort=None,
                                thinking_display='summarized', max_tokens=None,
                                temperature=None, top_p=None, top_k=None,
                                stop_sequences=None, response_format=None, tools=None,
                                tool_choice=None, stream=False, verbose=True,
                                host=None, local_timeout=None):
    """Query a Claude model through the Anthropic Messages API (the anthropic SDK) or OpenRouter.

    On OpenRouter the request is a chat completion with the unified `reasoning` parameter. On
    both hosts every request is streamed under the hood, so streamed and non-streamed calls
    return the same contract; `stream` only controls whether deltas are echoed live.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        A key of MODEL_CONFIG below.
    system_prompt : str or None
        System-level instruction, passed as the top-level `system` parameter.
    thinking : str, bool, or None
        'adaptive', 'enabled' (a fixed thinking_budget, for the models that support it),
        'disabled', or None for the model's default. True/False map to 'adaptive'/'disabled'.
        claude-fable-5-2 thinks on every request: 'disabled' and 'enabled' raise here because
        the API rejects both.
    thinking_budget : int or None
        Token budget for thinking='enabled' (minimum 1024).
    reasoning_effort : str or None
        Passed as output_config.effort (Anthropic) or reasoning.effort (OpenRouter). The valid
        levels are the model's 'effort_options'; None sends the model's default explicitly, which
        is the same as omitting it.
    thinking_display : str
        'summarized' (default) or 'omitted'. On a model that omits thinking text by default this
        decides whether reasoning_text is filled (OpenRouter: 'omitted' sends reasoning.exclude).
        Thinking is billed the same either way.
    max_tokens : int or None
        Output cap, thinking included. None -> 20000.
    temperature, top_p, top_k : float, float, int, or None
        Sampling parameters. Dropped (with a note when verbose) on a 'fixed_sampling' model,
        which rejects non-default values.
    stop_sequences : list[str] or None
        Custom stop sequences.
    response_format : dict or None
        Structured output, e.g. {"type": "json_schema", "schema": {...}}; sent as
        output_config.format, merged with the effort setting. Anthropic's own API only.
    tools : list, 'web search', or None
        Tool definitions; 'web search' enables the server-side web_search tool. Anthropic's own
        API only.
    tool_choice : dict or None
        Tool selection strategy, e.g. {"type": "auto"}. Anthropic's own API only.
    stream : bool
        Echo thinking and text deltas as they arrive.
    verbose : bool
        Print the response, usage and the cost estimate.
    host : None or 'OpenRouter'
        None -> Anthropic's own API (ANTHROPIC_API_KEY). 'OpenRouter' -> chat completions through
        OpenRouter (OPENROUTER_API_KEY) with require_parameters; only for a model with an
        'openrouter_name'.
    local_timeout : float or None
        Client-side wall-clock limit in seconds on the request, stream drain included
        (None = disabled). Raises TimeoutError; see _local_timeout.

    Returns
    -------
    dict
        The response contract described at the top of this module. If the stream ends without a
        final message, the reduced contract.

    Notes
    -----
    prompt_tokens includes cache reads and writes (Anthropic reports them separately from
    input_tokens), so cached_tokens is a subset of prompt_tokens, as in the other query
    functions. completion_tokens includes thinking on both hosts. Anthropic's own API does not
    report a separate thinking count (reasoning_tokens = 0) while OpenRouter does, so
    reasoning_tokens is not comparable between hosts.

    A refusal (stop_reason='refusal') comes back as an ordinary failed attempt: text='' (not
    None, so retry_llm_call does not retry it), reasoning_text=None, and no field marks the
    refusal: a refusal is treated like any other failure to attempt the task. For the same
    reason the API's server-side `fallbacks` option, which re-runs a declined request on another
    model, is never requested: the answer would come from a model other than the one recorded.

    Prices: 5-minute cache writes are 1.25x input and cache reads are far cheaper than input.
    1-hour cache writes, web search fees and regional surcharges are not included in
    cost_estimate. OpenRouter reports the billed cost, which is used when present.
    """

    # Prices in dollars per token.
    MODEL_CONFIG = {
        # Thinking is always on: the API accepts only adaptive (or no `thinking` field) and
        # rejects 'disabled' and budgets. Depth is set by effort alone. Sampling parameters and a
        # forced tool_choice are rejected. Thinking text is omitted by default and never raw.
        # Safety classifiers can end a request with stop_reason='refusal'.
        'claude-fable-5-2': {
            'openrouter_name': 'anthropic/claude-fable-5.2',
            'context_length': 1_000_000,
            'max_output': 128_000,
            'cost_in': 8.00 / 1e6,
            'cost_out': 40.00 / 1e6,
            'cost_cache_write_5m': 10.00 / 1e6,
            'cost_cache_read': 0.20 / 1e6,
            'default_thinking': 'adaptive',
            'supports_adaptive': True,
            'supports_extended': False,
            'supports_disable': False,  # the API rejects it
            'omits_thinking_text': True,
            'effort_options': ['low', 'medium', 'high', 'xhigh', 'max'],
            'default_effort': 'high',
            'fixed_sampling': True,
        },
    }

    THINKING_OPTIONS = [None, 'adaptive', 'enabled', 'disabled']
    DISPLAY_OPTIONS = ['summarized', 'omitted']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Resolve host ---
    allowed_hosts = [None, 'OpenRouter']
    if host not in allowed_hosts:
        raise ValueError(f'host `{host}` not in allowed hosts: `{allowed_hosts}`.')
    if host == 'OpenRouter' and not cfg.get('openrouter_name'):
        raise ValueError(f"Model '{model}' is not wired for OpenRouter; use host=None.")
    if host == 'OpenRouter' and (tools is not None or tool_choice is not None
                                 or response_format is not None):
        raise ValueError("tools/tool_choice/response_format are only wired on Anthropic's own "
                         "API (host=None).")

    # --- Validate thinking ---
    if isinstance(thinking, bool):
        thinking = 'adaptive' if thinking else 'disabled'
    if thinking not in THINKING_OPTIONS:
        raise ValueError(f"Invalid thinking {thinking!r}. Must be a bool or one of {THINKING_OPTIONS}")
    if thinking == 'disabled' and not cfg['supports_disable']:
        raise ValueError(f"Model '{model}' has no no-thinking arm. Use thinking='adaptive' (or None) "
                         f"and lower reasoning_effort instead.")
    if thinking is None:
        thinking = cfg['default_thinking']
    if thinking == 'adaptive' and not cfg['supports_adaptive']:
        raise ValueError(f"Model '{model}' does not support adaptive thinking; "
                         f"use thinking='enabled' (if supported) or None.")
    if thinking == 'enabled' and not cfg['supports_extended']:
        raise ValueError(f"Model '{model}' does not support extended ('enabled') thinking with a budget."
                         + (" Use thinking='adaptive' with reasoning_effort instead."
                            if cfg['supports_adaptive'] else ''))
    if thinking == 'enabled' and (thinking_budget is None or thinking_budget < 1024):
        raise ValueError("thinking='enabled' requires thinking_budget >= 1024.")
    if thinking_display not in DISPLAY_OPTIONS:
        raise ValueError(f"Invalid thinking_display {thinking_display!r}. Must be one of {DISPLAY_OPTIONS}")

    # --- Validate reasoning effort ---
    if reasoning_effort is None:
        reasoning_effort = cfg['default_effort']  # sent explicitly; same as omitting it
    if reasoning_effort is not None:
        if cfg['effort_options'] is None:
            raise ValueError(f"Model '{model}' does not support reasoning_effort.")
        if reasoning_effort not in cfg['effort_options']:
            raise ValueError(f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. "
                             f"Must be one of {cfg['effort_options']}")

    # --- Validate sampling parameters ---
    if cfg['fixed_sampling']:
        dropped = [name for name, value in (('temperature', temperature), ('top_p', top_p),
                                            ('top_k', top_k)) if value is not None]
        if dropped and verbose:
            print(f"Model '{model}' rejects non-default sampling parameters. Dropping {dropped}.")
        temperature = top_p = top_k = None
    if thinking in ('adaptive', 'enabled') and temperature not in (None, 1.0):
        raise ValueError(f"When thinking is '{thinking}', temperature must be unset or 1.0. "
                         f"Got temperature={temperature}. The API enforces this constraint.")

    # --- Validate max tokens ---
    if max_tokens is None:
        max_tokens = 20000
    if max_tokens > cfg['max_output']:
        raise ValueError(f"max_tokens={max_tokens} exceeds the {cfg['max_output']} output cap of '{model}'.")

    # --- Build API kwargs ---
    if host == 'OpenRouter':
        client = _get_async_client("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")

        messages = []
        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # Unified reasoning parameter: OpenRouter maps effort -> output_config.effort,
        # max_tokens -> thinking.budget_tokens, and returns summarized thinking by default.
        if thinking == 'adaptive':
            reasoning_cfg = {'enabled': True}
            if reasoning_effort is not None: reasoning_cfg['effort'] = reasoning_effort
        elif thinking == 'enabled':
            reasoning_cfg = {'max_tokens': thinking_budget}
        else:  # 'disabled' or None: no thinking
            reasoning_cfg = {'enabled': False}
        if thinking in ('adaptive', 'enabled') and thinking_display == 'omitted':
            reasoning_cfg['exclude'] = True  # not returned; still billed, as with display='omitted'

        # Build extra_body ONCE and only add keys (merge, never reassign).
        extra_body = {'provider': {'require_parameters': True}, 'reasoning': reasoning_cfg}
        if top_k is not None: extra_body['top_k'] = top_k

        kwargs = {
            'model': cfg['openrouter_name'],
            'messages': messages,
            'max_tokens': max_tokens,
            'stream': True,  # always streamed, as on Anthropic's own API
            'extra_body': extra_body,
        }
        if temperature is not None: kwargs['temperature'] = temperature
        if top_p is not None: kwargs['top_p'] = top_p
        if stop_sequences is not None: kwargs['stop'] = stop_sequences

    else:  # Anthropic's own API
        client = _get_anthropic_client("ANTHROPIC_API_KEY")

        kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if system_prompt is not None: kwargs['system'] = system_prompt
        if thinking == 'adaptive':
            kwargs['thinking'] = {'type': 'adaptive'}
            if cfg['omits_thinking_text']: kwargs['thinking']['display'] = thinking_display
        elif thinking == 'enabled':
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget}
        # thinking == 'disabled' or None: the field is omitted

        # effort and structured-output format share one output_config dict
        output_config = {}
        if reasoning_effort is not None: output_config['effort'] = reasoning_effort
        if response_format is not None: output_config['format'] = response_format
        if output_config: kwargs['output_config'] = output_config

        if temperature is not None: kwargs['temperature'] = temperature
        if top_p is not None: kwargs['top_p'] = top_p
        if top_k is not None: kwargs['top_k'] = top_k
        if stop_sequences is not None: kwargs['stop_sequences'] = stop_sequences
        if tools == 'web search':
            kwargs['tools'] = [{'type': 'web_search_20250305', 'name': 'web_search'}]
        elif tools is not None:
            kwargs['tools'] = tools
        if tool_choice is not None: kwargs['tool_choice'] = tool_choice

    # --- Call the API ---
    # Always streamed (OpenRouter: drained by the shared chat collector; Anthropic:
    # finalized with get_final_message()), so both modes share the extraction below and
    # a large max_tokens never trips a non-streaming duration guard. `stream` only
    # controls the live echo.
    echo = verbose and stream
    text_parts, reasoning_parts = [], []
    async with _local_timeout(local_timeout, f'{model} via {host or "Anthropic"}'):
        if host == 'OpenRouter':
            response = await client.chat.completions.create(**kwargs)
            collected = await _collect_chat_stream_async(response, verbose=echo)
        else:
            async with client.messages.stream(**kwargs) as stream_mgr:
                async for event in stream_mgr:
                    if event.type != 'content_block_delta':
                        continue
                    if event.delta.type == 'thinking_delta' and event.delta.thinking:
                        _stream_print_delta(event.delta.thinking, True, reasoning_parts, text_parts, echo)
                    elif event.delta.type == 'text_delta' and event.delta.text:
                        _stream_print_delta(event.delta.text, False, reasoning_parts, text_parts, echo)
                response = await stream_mgr.get_final_message()
    if echo and host != 'OpenRouter': print()  # the chat collector ends its own echo
    if verbose:
        print(f'response full: \n {response}')

    if host == 'OpenRouter':
        # --- Extract content ---
        if collected['usage'] is None:
            return collected  # interrupted stream: degrade to reduced contract
        text = collected['text'] or ''
        reasoning = collected['reasoning_text']
        if thinking_display == 'omitted':
            reasoning = None  # OpenRouter honors reasoning.exclude only on non-streamed requests
        usage = collected['usage']
        finish_reason = collected['finish_reason']
        response = collected['response']
        refused = finish_reason in ('content_filter', 'refusal')  # OpenRouter's normalized refusal

        # --- Extract usage and compute cost ---
        # OpenRouter's prompt_tokens already includes cache reads and writes, and
        # completion_tokens includes reasoning (reasoning_tokens is the informational subset).
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        pdetails = getattr(usage, 'prompt_tokens_details', None)
        cached_tokens = (getattr(pdetails, 'cached_tokens', 0) or 0) if pdetails else 0
        cache_write_tokens = (getattr(pdetails, 'cache_write_tokens', 0) or 0) if pdetails else 0
        cdetails = getattr(usage, 'completion_tokens_details', None)
        reasoning_tokens = (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0
        uncached_input = prompt_tokens - cached_tokens - cache_write_tokens
        server_provider = getattr(response, 'provider', None)

        truncated = (finish_reason == 'length')
        cost_actual = getattr(usage, 'cost', None)  # provider-reported cost wins when present

    else:  # Anthropic's own API
        # --- Extract content ---
        if response.stop_reason is None:
            # Stream ended without a terminal event; return what we have.
            return _stream_result(text_parts, reasoning_parts, response)
        text = ''.join(b.text for b in response.content if b.type == 'text')
        reasoning = '\n'.join(b.thinking for b in response.content
                              if b.type == 'thinking' and b.thinking) or None
        refused = (response.stop_reason == 'refusal')

        # --- Extract usage and compute cost ---
        # Anthropic reports input_tokens, cache writes, and cache reads as separate counts;
        # prompt_tokens is their sum, so cached_tokens is a subset of it as elsewhere.
        usage = response.usage
        uncached_input = usage.input_tokens
        cached_tokens = usage.cache_read_input_tokens or 0
        cache_write_tokens = usage.cache_creation_input_tokens or 0
        prompt_tokens = uncached_input + cached_tokens + cache_write_tokens
        completion_tokens = usage.output_tokens  # includes thinking
        reasoning_tokens = 0  # Anthropic does not report thinking tokens separately
        total_tokens = prompt_tokens + completion_tokens
        server_provider = 'Anthropic'

        truncated = (response.stop_reason == 'max_tokens')
        cost_actual = None

    # --- Shared post-processing ---
    if refused:
        # A refusal is scored like any other no-attempt failure: empty text (not None,
        # so retry_llm_call does not retry it) and nothing that marks it as a refusal.
        text, reasoning = '', None

    status = 'incomplete' if truncated else 'completed'
    incomplete_reason = 'max_tokens' if truncated else None

    tier = '_long' if prompt_tokens > cfg.get('long_context_threshold', math.inf) else ''
    cost_est = (
            uncached_input * cfg['cost_in' + tier]
            + cached_tokens * cfg['cost_cache_read' + tier]
            + cache_write_tokens * cfg['cost_cache_write_5m' + tier]
            + completion_tokens * cfg['cost_out' + tier]
    )
    cost = cost_est if cost_actual is None else cost_actual

    if verbose:
        if not stream:  # streamed output was already echoed
            if reasoning:
                print(f'.--- Reasoning ---.\n {reasoning} \n\n')
            print(f'.--- Response ---.\n {text}')
        print(f"\nTokens — input: {prompt_tokens} (cached: {cached_tokens}, "
              f"cache writes: {cache_write_tokens}), "
              f"output: {completion_tokens} (incl. thinking), "
              f"total: {total_tokens}, server provider: {server_provider}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': text,
        'reasoning_text': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'cache_write_tokens': cache_write_tokens,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
    }

async def google_query_async(prompt, model='gemini-4-carbon', thinking=False, thinking_budget=None,
                             reasoning_effort=None, system_prompt=None, tool=None, max_tokens=None,
                             temperature=None, top_p=None, top_k=None, response_format=None,
                             flex=True, stream=False, verbose=True, host=None,
                             local_timeout=None):
    """Query a Google Gemini model through the Gemini API (google-genai SDK) or OpenRouter.

    The Gemini API branch uses the google-genai SDK, because thinking levels, per-part thought
    flags, thought summaries and the flex service tier are not all reachable through an
    OpenAI-compatible layer. The request configuration is passed as plain dicts, which the SDK
    coerces. The OpenRouter branch is a chat completion with the unified `reasoning` parameter.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        A key of MODEL_CONFIG below.
    thinking : bool
        Whether thought summaries are requested (Gemini API). A model that controls thinking by
        level cannot turn it off; a reasoning_effort sets thinking to True.
    thinking_budget : int or None
        Not accepted by a model that controls thinking by level (raises): use reasoning_effort.
    reasoning_effort : str or None
        One of the model's 'thinking_levels'. None defers to the server's default level. Sent as
        thinking_level (Gemini API) or reasoning.effort (OpenRouter).
    system_prompt : str or None
        Gemini API: the `system_instruction` field of the generation config. OpenRouter: a system
        message.
    tool : str or None
        'web_search' enables Grounding with Google Search (Gemini API only; raises on
        OpenRouter). Grounding fees are not included in cost_estimate.
    max_tokens : int or None
        Output cap. Thought tokens count against it, so with a low cap a model can spend all of
        it on thoughts and return an empty text with incomplete_reason 'max_output_tokens'.
        On a 'long_decode' model (gemini-4-carbon) one request returns at most
        'max_output_per_request' tokens of thinking and answer, and max_tokens is the total
        budget over several requests. A reply that needs more is continued automatically: by the
        SDK on the Gemini API, by a loop here on OpenRouter. The returned text and reasoning are
        joined over the requests; the token counts and the cost are summed over them, and each
        request bills its full prompt again, so prompt_tokens can be several times the prompt's
        length. A max_tokens at or below the per-request cap (or None) leaves a single request.
    temperature, top_p, top_k : float / float / int or None
        Sampling parameters; None defers to the server defaults. top_k is sent in extra_body on
        OpenRouter.
    response_format : dict or None
        {'type': 'json_object'} maps to response_mime_type='application/json' on the Gemini API
        and is passed through as given on OpenRouter.
    flex : bool
        Request the flex service tier (default): half the standard token rates, served on spare
        capacity with variable latency and best-effort availability. At capacity the API answers
        503 or 429 and never upgrades the request to the standard tier, so retries are the
        caller's business. Sent as service_tier='flex' on both hosts.
    stream : bool
        Stream the response. Both hosts return the full contract; the reduced one only if the
        stream ends without usage data.
    verbose : bool
        Print the response text, thought summaries, usage and the cost estimate.
    host : None or 'OpenRouter'
        None -> the Gemini API (GOOGLE_API_KEY). 'OpenRouter' -> chat completions through
        OpenRouter (OPENROUTER_API_KEY) with require_parameters.
    local_timeout : float or None
        Client-side wall-clock limit in seconds on each request, stream drain included
        (None = disabled). Raises TimeoutError; see _local_timeout.

    Returns
    -------
    dict
        The response contract described at the top of this module. On the Gemini API
        'completion_tokens' is the visible answer only, with thoughts counted separately in
        'reasoning_tokens'; on OpenRouter completion_tokens includes the reasoning tokens. The
        cost follows each convention, but completion_tokens is not comparable between hosts.
        'reasoning_text' on the Gemini API holds thought summaries (never the raw reasoning),
        and only when thinking is on.

    Notes
    -----
    No safety-threshold override is sent. Input-side blocks and SAFETY or RECITATION finish
    reasons surface through 'status' and 'incomplete_reason'.

    Thought tokens bill at the output rate. Implicit prefix caching is automatic and reported in
    cached_tokens; the flex discount is assumed to apply to cache reads as well.
    """

    # gemini-4-carbon launched at half its list price; the list price applies from 2026-12-01.
    carbon_price_factor = 1 if datetime.date.today() < datetime.date(2026, 12, 1) else 2

    # Prices in dollars per token.
    MODEL_CONFIG = {
        # Thinking levels low/medium/high (it cannot be turned off; the server default is
        # 'medium'). Long decode: one request returns at most 'max_output_per_request' tokens of
        # thinking and answer, and max_tokens is the total budget over several requests ("hops"),
        # up to the context window minus the input ('max_output'). On the Gemini API the SDK
        # (google-genai >= 2.30.1) continues automatically; on OpenRouter this function sends the
        # continuation token back itself. Flex is half price on every rate.
        'gemini-4-carbon': {
            'gemini_name': 'gemini-4-carbon',
            'openrouter_name': 'google/gemini-4-carbon',
            'context_length': 1_048_576,
            'max_output': 1_048_576,
            'max_output_per_request': 65_536,
            'long_decode': True,
            'thinking_api': 'level',
            'thinks_by_default': True,
            'thinking_levels': ['low', 'medium', 'high'],
            'supports_flex': True,
            'cost_in': carbon_price_factor * 2.00 / 1e6,
            'cost_cached': carbon_price_factor * 0.10 / 1e6,
            'cost_out': carbon_price_factor * 10.00 / 1e6,
            'cost_in_flex': carbon_price_factor * 1.00 / 1e6,
            'cost_cached_flex': carbon_price_factor * 0.05 / 1e6,
            'cost_out_flex': carbon_price_factor * 5.00 / 1e6,
        },
    }

    TOOL_OPTIONS = ['web_search']

    # --- Validate model ---
    if model not in MODEL_CONFIG:
        raise ValueError(f"Invalid model '{model}'. Must be one of {list(MODEL_CONFIG)}")
    cfg = MODEL_CONFIG[model]

    # --- Resolve host ---
    allowed_hosts = [None, 'OpenRouter']
    if host not in allowed_hosts:
        raise ValueError(f'host `{host}` not in allowed hosts: `{allowed_hosts}`.')
    if host is None and cfg['gemini_name'] is None:
        raise ValueError(f"Model '{model}' is not served by the Gemini API; use host='OpenRouter'.")
    if host == 'OpenRouter' and cfg['openrouter_name'] is None:
        raise ValueError(f"Model '{model}' has no OpenRouter listing; use host=None.")

    # --- Reconcile the thinking toggle with the level ---
    if thinking_budget is not None:
        raise ValueError(
            f"Model '{model}' controls thinking with reasoning_effort "
            f"{cfg['thinking_levels']}, not thinking_budget."
        )
    if reasoning_effort is not None:
        if reasoning_effort not in cfg['thinking_levels']:
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}'. "
                f"Must be one of {cfg['thinking_levels']}"
            )
        if not thinking and reasoning_effort != 'minimal':
            print(f"Conflicting thinking=False and reasoning_effort='{reasoning_effort}'. "
                  f"Overriding thinking to True.")
            thinking = True

    # --- Validate tool ---
    if tool is not None and tool not in TOOL_OPTIONS:
        raise ValueError(f"Invalid tool '{tool}'. Must be one of {TOOL_OPTIONS}")
    if tool is not None and host == 'OpenRouter':
        raise ValueError(
            "Grounding with Google Search is only wired on the Gemini API (host=None); "
            "OpenRouter does not forward it."
        )

    # --- Build API kwargs ---
    if host == 'OpenRouter':
        client = _get_async_client("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")

        messages = []
        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # extra_body is built once and only added to, so that require_parameters is never lost.
        extra_body = {
            'provider': {'require_parameters': True},
            'order': ['google-ai-studio/flex'],
            'ignore': ['google-vertex/global/flex', 'google-vertex/global', 'google-vertex',]
        }
        # Thinking cannot be turned off; with no level given, nothing is sent and the provider's
        # default applies.
        if reasoning_effort is not None:
            extra_body['reasoning'] = {'effort': reasoning_effort}
        if top_k is not None:
            extra_body['top_k'] = top_k

        kwargs = {
            'model': cfg['openrouter_name'],
            'messages': messages,
            'stream': stream,
            'extra_body': extra_body,
        }
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        if temperature is not None:
            kwargs['temperature'] = temperature
        if top_p is not None:
            kwargs['top_p'] = top_p
        if response_format is not None:
            kwargs['response_format'] = response_format
        if flex == True:
            kwargs['service_tier'] = 'flex'

    else:  # the Gemini API
        client = _get_genai_client("GOOGLE_API_KEY")

        # GenerateContentConfig as a plain dict, which the SDK coerces.
        gen_config = {}
        if system_prompt is not None:
            gen_config['system_instruction'] = system_prompt
        if max_tokens is not None:
            gen_config['max_output_tokens'] = max_tokens
        if temperature is not None:
            gen_config['temperature'] = temperature
        if top_p is not None:
            gen_config['top_p'] = top_p
        if top_k is not None:
            gen_config['top_k'] = top_k
        if response_format is not None:
            if isinstance(response_format, dict) and response_format.get('type') == 'json_object':
                gen_config['response_mime_type'] = 'application/json'
            else:
                raise ValueError(
                    "Only {'type': 'json_object'} is wired on the Gemini API; "
                    "schema-constrained output is not implemented here."
                )
        if tool == 'web_search':
            gen_config['tools'] = [{'google_search': {}}]
        if flex == True:
            # A top-level request field, which the SDK takes from the config. At capacity flex
            # answers 503 or 429 with no upgrade to the standard tier; retries are the caller's.
            gen_config['service_tier'] = 'flex'

        tc = {}
        if thinking:
            tc['include_thoughts'] = True
        if reasoning_effort is not None:
            tc['thinking_level'] = reasoning_effort
        if tc:
            gen_config['thinking_config'] = tc

        kwargs = {
            'model': cfg['gemini_name'],
            'contents': prompt,
            'config': gen_config,
        }

    # --- Call the API ---
    if host == 'OpenRouter':
        # One request per hop. A long-decode model (cfg['long_decode']) stops each request at its
        # per-request output cap and hands back a continuation token; the loop sends that token
        # back until the model finishes or the max_tokens budget is spent. Every other model, and
        # every reply that fits in one request, is a single pass through the loop.
        hops, interrupted = [], False
        while True:
            async with _local_timeout(local_timeout, f'{model} via {host}'):
                response = await client.chat.completions.create(**kwargs)
                collected = await _collect_chat_stream_async(response, verbose=verbose) if stream else None
            hop = _openrouter_chat_hop(response, collected)
            if hop['usage'] is None:  # interrupted stream
                if not hops:
                    return collected  # degrade to reduced contract
                interrupted = True  # keep what the earlier hops returned
                break
            hops.append(hop)
            spent = sum(h['usage'].completion_tokens for h in hops)
            if not (cfg.get('long_decode') and len(hops) < MAX_CONTINUATION_HOPS
                    and _gemini_hop_continues(hop, spent, max_tokens)):
                break
            if verbose:
                print(f'\n[hop {len(hops)} stopped at the per-request output cap after {spent} output '
                      f'tokens in total; continuing]\n')
            # The same conversation plus an empty assistant turn that carries only the latest token.
            # The text of earlier hops is not resent.
            kwargs['messages'] = messages + [{'role': 'assistant', 'content': '',
                                              'reasoning_details': [hop['continuation_item']]}]
    else:
        async with _local_timeout(local_timeout, f'{model} via Google'):
            if stream:
                response = await client.aio.models.generate_content_stream(**kwargs)
                collected = await _collect_genai_stream_async(response, verbose=verbose)
            else:
                response = await client.aio.models.generate_content(**kwargs)

    if verbose == True:
        print(f'full response: \n {response}\n\n')

    # --- Extract content ---
    if host == 'OpenRouter':
        # One hop is the usual case; several hops (long decode) are joined in order.
        if len(hops) == 1:
            output_text, reasoning = hops[0]['text'], hops[0]['reasoning_text']
        else:
            output_text = ''.join(h['text'] or '' for h in hops)
            reasoning = ''.join(h['reasoning_text'] or '' for h in hops) or None
        finish_reason = hops[-1]['finish_reason']
        server_provider = hops[-1]['server_provider']
        # --- Extract usage and compute cost ---
        # Usage is reported per hop and each hop is billed separately, its full prompt included,
        # so every count below is the sum over hops (with one hop: that hop's own figures).
        # completion_tokens already includes the reasoning tokens on OpenRouter, so they are not
        # added again. The estimate uses the standard rates; the cost the provider reports wins
        # when it is present.
        prompt_tokens = completion_tokens = total_tokens = reasoning_tokens = cached_tokens = 0
        cost = 0.0
        for hop in hops:
            usage = hop['usage']
            cdetails = getattr(usage, 'completion_tokens_details', None)
            pdetails = getattr(usage, 'prompt_tokens_details', None)
            hop_cached = (getattr(pdetails, 'cached_tokens', 0) or 0) if pdetails else 0
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            total_tokens += usage.total_tokens
            reasoning_tokens += (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0
            cached_tokens += hop_cached
            cost_est = ((usage.prompt_tokens - hop_cached) * cfg['cost_in']
                        + hop_cached * cfg['cost_cached']
                        + usage.completion_tokens * cfg['cost_out'])
            cost_actual = getattr(usage, 'cost', None)
            cost += cost_est if cost_actual is None else cost_actual

        if interrupted:  # a continuation hop's stream ended without usage; earlier hops are kept
            status, incomplete_reason = 'incomplete', 'continuation_interrupted'
        else:
            status = 'incomplete' if finish_reason == 'length' else 'completed'
            incomplete_reason = 'max_output_tokens' if finish_reason == 'length' else None

    else:  # the Gemini API
        # A stream arrives as chunks: text parts are increments, while finish_reason,
        # prompt_feedback, and the cumulative usage_metadata are read from the last chunk
        # that carries each. A non-streamed response is the one-chunk case of the same walk.
        if stream:
            chunks = collected['chunks']
            if not any(getattr(c, 'usage_metadata', None) for c in chunks):
                return collected  # interrupted stream: degrade to reduced contract
            response = chunks[-1]
        else:
            chunks = [response]

        # --- Extract content (typed parts; thought parts are summaries) ---
        finish_reason, pf, um = None, None, None
        output_parts, thought_parts = [], []
        for chunk in chunks:
            if chunk.candidates:
                cand = chunk.candidates[0]
                fr = getattr(cand, 'finish_reason', None)
                if fr is not None:
                    finish_reason = getattr(fr, 'name', None) or str(fr)
                parts = (cand.content.parts if (cand.content and cand.content.parts) else [])
                for part in parts:
                    piece = getattr(part, 'text', None)
                    if piece:
                        (thought_parts if getattr(part, 'thought', False) else output_parts).append(piece)
            pf = getattr(chunk, 'prompt_feedback', None) or pf
            um = getattr(chunk, 'usage_metadata', None) or um

        output_text = ''.join(output_parts)
        reasoning = ''.join(thought_parts) if thought_parts else None

        # Input-side safety blocks arrive with NO candidates + prompt_feedback.
        block_reason = getattr(pf, 'block_reason', None) if pf is not None else None

        if finish_reason == 'STOP':
            status = 'completed'
            incomplete_reason = None
        elif finish_reason == 'MAX_TOKENS':
            # Thoughts count against max_tokens, so this can arrive with text == '' (the whole
            # cap spent on thoughts). Callers should read status, not len(text).
            status = 'incomplete'
            incomplete_reason = 'max_output_tokens'
        else:
            status = 'incomplete'
            if block_reason is not None:
                incomplete_reason = getattr(block_reason, 'name', None) or str(block_reason)
            else:
                incomplete_reason = finish_reason  # SAFETY / RECITATION / ...

        # --- Extract usage and compute cost ---
        prompt_tokens = (getattr(um, 'prompt_token_count', 0) or 0) if um else 0
        completion_tokens = (getattr(um, 'candidates_token_count', 0) or 0) if um else 0  # visible answer only
        reasoning_tokens = (getattr(um, 'thoughts_token_count', 0) or 0) if um else 0
        cached_tokens = (getattr(um, 'cached_content_token_count', 0) or 0) if um else 0  # implicit caching
        total_tokens = (getattr(um, 'total_token_count', 0) or 0) if um else 0

        # cached_content_token_count is a subset of prompt_token_count.
        # Thoughts bill at the output rate on both tiers.
        non_cached_input = prompt_tokens - cached_tokens
        if flex == False:
            cost = (non_cached_input * cfg['cost_in']
                    + cached_tokens * cfg['cost_cached']
                    + (completion_tokens + reasoning_tokens) * cfg['cost_out'])
        else:
            cost = (non_cached_input * cfg['cost_in_flex']
                    + cached_tokens * cfg['cost_cached_flex']
                    + (completion_tokens + reasoning_tokens) * cfg['cost_out_flex'])

        server_provider = 'Google'

    # --- Shared post-processing ---
    if verbose:
        if reasoning:
            print(".--- Reasoning ---.")
            print(reasoning)
            print("\n.--- Answer ---.")
        print(output_text)
        print(f"\nTokens -- prompt: {prompt_tokens} (cached: {cached_tokens}), "
              f"completion: {completion_tokens}, reasoning: {reasoning_tokens}, "
              f"total: {total_tokens}, server provider: {server_provider}")
        print(f"Estimated cost: ${cost:.6f}")

    return {
        'text': output_text,
        'reasoning_text': reasoning,
        'prompt_tokens': prompt_tokens,
        'cached_tokens': cached_tokens,
        'cache_write_tokens': 0,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'cost_estimate': cost,
        'response': response,
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
    }


# ------------------------------------------------------------------- the local chat-query loop
# One chat round against the local server, then optional continuation rounds. The scoring code
# uses neither option (a tiny model's answer is whatever one request returns), so they are off by
# default; they exist for answers longer than one request can hold:
#   auto_continue   resumes a round that stopped on finish_reason='length'
#   resume_on_drop  resumes a round whose stream dropped mid-body (STREAM_DROP_ERRORS)
# How a round is resumed is set by continue_mode:
#   'default'    -- system prompt + one user turn: continue_template around the partial. The
#                   original prompt is not resent.
#   'multi-turn' -- original messages + the partial as the assistant's turn + a user turn asking
#                   it to resume (DEFAULT_MULTITURN_CONTINUE_MESSAGE).
#   'raw'        -- true continuation: the chat template is rendered locally (the model's
#                   tokenizer), the partial appended as it is with the reasoning block still
#                   open, and the whole thing sent to /completions, so the model resumes
#                   mid-sequence as if never cut.

DEFAULT_CONTINUE_TEMPLATE = (
        "{partial}\n\n"
        "[The text above is your own unfinished work, cut off by an output-token limit. "
        "Resume from exactly where it stops. Do not restart, re-derive, or summarize steps "
        "already completed. When you are done, state the final answer.]"
    )

DEFAULT_DROP_CONTINUE_TEMPLATE = (
        "{partial}\n\n"
        "[The text above is your own unfinished work, interrupted partway through. "
        "Resume from exactly where it stops. Do not restart, re-derive, or summarize steps "
        "already completed. When you are done, state the final answer.]"
    )

# continue_mode='multi-turn': the partial goes back as the assistant's own turn, then this user turn
DEFAULT_MULTITURN_CONTINUE_MESSAGE = (
        "[Your previous response was interrupted partway through. Resume from exactly where it stops. "
        "Do not restart, re-derive, or summarize steps already completed. When you are done, state the "
        "final answer.]"
    )

CONTINUE_MODES = ('default', 'raw', 'multi-turn')

THINK_TAGS = ('<think>', '</think>')

# Mid-stream transport failures that leave a usable partial. httpx raises these raw while a
# stream is being iterated; the SDK only wraps errors raised before the response starts.
STREAM_DROP_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError)

class StreamInterruptedError(RuntimeError):
    """A streamed response ended cleanly (HTTP 200, no exception) before it finished: either no
    chunk at all, or some text but no finish_reason and no usage chunk, with nothing to resume it.
    This happens when a request reaches a server whose engine is still loading, and the next
    attempt usually succeeds. Raised so the callers' retry ladders (retry_llm_call, the scoring
    workers) treat it like any transient failure instead of hitting KeyError on the reduced
    contract."""

_hf_tokenizers = {}

def _get_hf_tokenizer(repo, revision=None):
    key = (repo, revision)
    if key not in _hf_tokenizers:
        from transformers import AutoTokenizer  # heavy import; only continue_mode='raw' needs it
        _hf_tokenizers[key] = AutoTokenizer.from_pretrained(repo, revision=revision)
    return _hf_tokenizers[key]

class _RawContinuation:
    """The assistant sequence so far, verbatim with think tags, for continue_mode='raw'."""

    def __init__(self, repo, messages, template_kwargs=None, think_tags=THINK_TAGS, revision=None):
        self.tok = _get_hf_tokenizer(repo, revision)
        self.base = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 **(template_kwargs or {}))
        self.open, self.close = think_tags
        self.opened_by_template = self.base.rstrip().endswith(self.open)  # some templates open the block
        self.thinking = self.opened_by_template
        self.text = ''

    def count(self, text):
        return len(self.tok(text, add_special_tokens=False)['input_ids']) if text else 0

    def absorb_chat(self, reasoning, text):
        """Round 1 is a normal chat call: rebuild the raw sequence from the host's parsed fields."""
        if reasoning:
            self.thinking = True
            self.text += reasoning if not text else (
                reasoning.rstrip('\n') + '\n' + self.close + '\n\n' + text.lstrip('\n'))
        else:
            self.text += text or ''  # also covers hosts that leave the reasoning inline in content
        if self.text.lstrip().startswith(self.open):
            self.thinking = True

    def absorb_raw(self, new):
        """Append a /completions round verbatim; return that round's (reasoning, text) split."""
        in_reasoning = self.thinking and self.close not in self.text
        self.text += new
        if not in_reasoning:
            return '', new
        reasoning, _, text = new.partition(self.close)
        return reasoning, text.lstrip('\n')

    def trim(self):
        """Cut back to the last whitespace so a resume never starts inside a re-tokenized word."""
        cut = max(self.text.rfind(' '), self.text.rfind('\n'))
        if cut > 0:
            self.text = self.text[:cut]

    def prompt(self):
        lead = (self.open + '\n' if self.thinking and not self.opened_by_template
                and not self.text.lstrip().startswith(self.open) else '')
        return self.base + lead + self.text

    def split(self):
        """Final (reasoning_text, text) of the whole sequence."""
        if self.close in self.text:
            reasoning, _, text = self.text.partition(self.close)
        elif self.thinking:
            reasoning, text = self.text, ''
        else:
            return None, self.text.strip()
        reasoning = reasoning.strip()
        if reasoning.startswith(self.open):
            reasoning = reasoning[len(self.open):].strip()
        return reasoning or None, text.strip()

    def request(self, chat_kwargs, context_window=None):
        """/completions kwargs mirroring the chat request: same sampling, prompt for messages.

        max_tokens is always explicit and capped at the context room left after the prompt:
        unlike chat, /completions does not size an omitted max_tokens to the remaining context
        (a server may fill in the whole window and reject the request, or apply a default of 16
        tokens)."""
        req = {k: v for k, v in chat_kwargs.items()
               if k not in ('messages', 'tools', 'prompt_cache_key', 'extra_body', 'stream_options')}
        req['prompt'] = self.prompt()
        if context_window:
            room = max(context_window - self.count(req['prompt']) - 1, 1)
            req['max_tokens'] = room if req.get('max_tokens') is None else min(req['max_tokens'], room)
        if req.get('stream'):
            req['stream_options'] = {'include_usage': True}
        # chat-only fields; chat_template_kwargs were already applied by the local render
        extra = {k: v for k, v in (chat_kwargs.get('extra_body') or {}).items()
                 if k not in ('chat_template_kwargs', 'reasoning')}
        if extra:
            req['extra_body'] = extra
        return req

def _continue_messages(mode, messages, carry, template):
    """Chat messages for the next round under continue_mode 'default' or 'multi-turn'."""
    if mode == 'multi-turn':
        return messages + [{'role': 'assistant', 'content': carry},
                           {'role': 'user', 'content': DEFAULT_MULTITURN_CONTINUE_MESSAGE}]
    return [m for m in messages if m['role'] == 'system'] + [
        {'role': 'user', 'content': template.format(partial=carry)}]

def _usage_counts(usage):
    """(prompt, cached, completion, reasoning, total, billed cost or None) from a usage object."""
    cdetails = getattr(usage, 'completion_tokens_details', None)
    pdetails = getattr(usage, 'prompt_tokens_details', None)
    reasoning = (getattr(cdetails, 'reasoning_tokens', 0) or 0) if cdetails else 0
    cached = (getattr(pdetails, 'cached_tokens', 0) or 0) if pdetails else 0
    cached = cached or getattr(usage, 'cached_tokens', 0) or 0  # some servers: top level of usage
    billed = getattr(usage, 'cost', None) or getattr(usage, 'estimated_cost', None)
    return (usage.prompt_tokens, cached, usage.completion_tokens, reasoning, usage.total_tokens, billed)

async def _chat_query_loop(client, kwargs, messages, *, stream, verbose, label, local_timeout,
                           rates, auto_continue, max_rounds, context_budget, continue_template,
                           continue_include_reasoning, continue_carry, continue_mode,
                           resume_on_drop, max_drop_resumes, drop_resume_wait, raw_source,
                           context_window=None):
    """One chat round, then continuation rounds while auto_continue sees finish_reason='length'
    or resume_on_drop sees a mid-stream drop. `rates` = (cost_in, cost_out, cost_in_hit) per
    token, or None (no cost). `raw_source` = (hf_repo, revision, think_tags) for 'raw'.
    `context_window` = the host's total context (prompt + output) in tokens, used to size
    /completions requests in 'raw' mode; required there when max_tokens is None.

    Returns the aggregate fields ('text', 'reasoning_text', token counts, 'cost', 'response',
    'finish_reason', 'rounds', 'drop_resumes', 'usage_estimated'). A stream that closes before
    finishing (no finish_reason, no usage) is resumed under resume_on_drop when it carried text,
    raises StreamInterruptedError on round 1 otherwise, and on a later round keeps the earlier
    rounds with finish_reason 'dropped'. A finished stream without a usage chunk keeps its text
    with estimated token counts (usage_estimated=True)."""
    if continue_mode not in CONTINUE_MODES:
        raise ValueError(f'continue_mode must be one of {CONTINUE_MODES}.')
    if continue_carry not in ('all', 'last'):
        raise ValueError("continue_carry must be 'all' or 'last'.")
    raw = None
    if continue_mode == 'raw' and (auto_continue or resume_on_drop):
        if kwargs.get('tools'):
            raise ValueError("continue_mode='raw' does not support tools.")
        repo, revision, think_tags = raw_source
        if not repo:
            raise ValueError("continue_mode='raw' needs the model's Hugging Face repo ('hf_repo').")
        if kwargs.get('max_tokens') is None and not context_window:
            raise ValueError("continue_mode='raw' with max_tokens=None needs the host's context window "
                             "('max_model_len'): /completions does not size an omitted max_tokens "
                             "to the remaining context.")
        template_kwargs = (kwargs.get('extra_body') or {}).get('chat_template_kwargs')
        raw = _RawContinuation(repo, messages, template_kwargs, think_tags, revision)

    base_max = kwargs.get('max_tokens')
    text_parts, reasoning_parts, partials = [], [], []
    carry_tokens = 0  # approx. tokens in the carried partial, for the context guard
    agg = {'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0,
           'reasoning_tokens': 0, 'total_tokens': 0, 'cost': 0.0}
    rounds = drop_resumes = 0
    base_prompt_tokens = None
    usage_estimated = False
    last_response = None

    while True:
        rounds += 1
        raw_round = raw is not None and rounds > 1  # round 1 is always a plain chat call
        dropped = None
        if raw_round:
            request = raw.request(kwargs, context_window)
            sent = request['prompt']
        else:
            sent = '\n'.join(m['content'] for m in kwargs['messages'] if isinstance(m.get('content'), str))
        try:
            async with _local_timeout(local_timeout, f'{label} (round {rounds})'):
                if raw_round:
                    response = await client.completions.create(**request)
                    if stream:
                        collected = await _collect_completion_stream_async(response, verbose=verbose)
                else:
                    response = await client.chat.completions.create(**kwargs)
                    if stream:
                        collected = await _collect_chat_stream_async(response, verbose=verbose)
        except STREAM_DROP_ERRORS as exc:
            partial = getattr(exc, 'stream_partial', None)
            if (resume_on_drop and drop_resumes < max_drop_resumes and partial
                    and (partial.get('text') or partial.get('reasoning_text'))):
                dropped = partial
            if dropped is None:
                raise
            drop_resumes += 1
            n_chars = len(dropped.get('text') or '') + len(dropped.get('reasoning_text') or '')
            print(f'resume_on_drop: {label} round {rounds} stream dropped after {n_chars} chars '
                  f'({ut.describe_exc(exc)}); resuming via continue_mode={continue_mode!r} '
                  f'({drop_resumes}/{max_drop_resumes})', flush=True)

        if dropped is None and stream and collected['usage'] is None and collected['finish_reason'] is None:
            # The stream closed cleanly (HTTP 200, no exception) before finishing, as happens
            # when the engine is still loading. Something arrived -> resume it like a drop under
            # resume_on_drop; otherwise a retryable error on round 1, or keep the earlier rounds
            # on a later round.
            n_chars = len(collected.get('text') or '') + len(collected.get('reasoning_text') or '')
            if resume_on_drop and drop_resumes < max_drop_resumes and n_chars:
                dropped = collected
                drop_resumes += 1
                print(f'resume_on_drop: {label} round {rounds} stream closed unfinished after '
                      f'{n_chars} chars; resuming via continue_mode={continue_mode!r} '
                      f'({drop_resumes}/{max_drop_resumes})', flush=True)
            elif rounds == 1:
                raise StreamInterruptedError(
                    f'{label}: stream closed ' + ('without sending any chunk' if n_chars == 0 else
                                                  f'unfinished after {n_chars} chars (not resumed)'))
            else:
                print(f'{label}: round {rounds} stream closed unfinished after {n_chars} chars; '
                      f'keeping the earlier rounds, output marked dropped.', flush=True)
                collected['finish_reason'] = 'dropped'

        if dropped is not None:
            new_text, reasoning = dropped.get('text') or '', dropped.get('reasoning_text')
            usage, finish_reason = None, 'dropped'
            last_response = dropped.get('response') or last_response
        elif stream:
            # usage may be None when the host finished the stream but omitted the usage chunk:
            # the answer is kept and the token counts are estimated below (usage_estimated=True)
            new_text, reasoning = collected['text'], collected['reasoning_text']
            usage, finish_reason = collected['usage'], collected['finish_reason']
            last_response = collected['response']
        else:
            if verbose:
                print(f'response full: {response}')
            if raw_round:
                new_text, reasoning = response.choices[0].text, None
            else:
                message = response.choices[0].message
                new_text = message.content
                reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
            usage = response.usage
            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            last_response = response

        if raw_round:
            reasoning, output_text = raw.absorb_raw(new_text or '')
        else:
            output_text = new_text
            if raw is not None:
                raw.absorb_chat(reasoning, output_text)

        # --- per-round usage / cost ---
        if usage is not None:
            (prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens,
             total_tokens, billed) = _usage_counts(usage)
            if raw_round and not reasoning_tokens:  # /completions does not split reasoning out
                reasoning_tokens = raw.count(reasoning)
        else:
            # a dropped stream never delivers its usage chunk: estimate (tokenizer if loaded, else chars/4)
            usage_estimated = True
            count = raw.count if raw is not None else (lambda t: len(t) // 4 if t else 0)
            prompt_tokens, cached_tokens = count(sent), 0
            reasoning_tokens = count(reasoning or '')
            completion_tokens = reasoning_tokens + count(output_text or '')
            total_tokens, billed = prompt_tokens + completion_tokens, None
        if rates is None:
            cost = 0.0
        else:
            cost_in, cost_out, cost_in_hit = rates
            # cached tokens are a subset of prompt_tokens: priced at the hit rate, excluded from the full rate
            cost_est = (cached_tokens * cost_in_hit + (prompt_tokens - cached_tokens) * cost_in
                        + completion_tokens * cost_out)
            cost = cost_est if billed is None else billed

        text_parts.append(output_text or '')
        reasoning_parts.append(reasoning or '')
        agg['prompt_tokens'] += prompt_tokens
        agg['cached_tokens'] += cached_tokens
        agg['completion_tokens'] += completion_tokens
        agg['reasoning_tokens'] += reasoning_tokens
        agg['total_tokens'] += total_tokens
        agg['cost'] += cost
        if base_prompt_tokens is None:
            base_prompt_tokens = prompt_tokens

        if verbose:
            if reasoning:
                print(".--- Reasoning ---.\n")
                print(reasoning)
                print("\n.--- Answer ---.\n")
            print(output_text)
            print(f"\nTokens -- prompt: {prompt_tokens} (cached: {cached_tokens}), "
                  f"reasoning: {reasoning_tokens}, completion: {completion_tokens}, total: {total_tokens}"
                  + (' [estimated]' if usage is None else ''))
            if rates is not None:
                print(f"Cost: ${cost:.6f}")

        # build this round's partial once; used both for carry and for the guard
        partial = ''
        if continue_include_reasoning and reasoning:
            partial += reasoning
        if output_text:
            partial += ('\n' if partial else '') + output_text
        partials.append(partial)
        cumulative = continue_carry == 'all' or raw is not None  # raw always resends everything
        carry_tokens = (carry_tokens + completion_tokens) if cumulative else completion_tokens

        # a drop and its resume share one round's max_tokens, so totals match an uncut round
        resume_budget = None
        if dropped is not None and kwargs.get('max_tokens') is not None:
            resume_budget = kwargs['max_tokens'] - completion_tokens
            if resume_budget <= 0:  # the budget was spent anyway: treat it as a length stop
                dropped, finish_reason = None, 'length'
        if dropped is not None:
            reason = 'drop'
        elif auto_continue and finish_reason == 'length':
            reason = 'length'
        else:
            break

        # next round's prompt ~= (original prompt, if resent) + carried partial(s) + ~100 template tokens
        resent = base_prompt_tokens if continue_mode in ('raw', 'multi-turn') else 0
        next_prompt_est = resent + carry_tokens + 100
        next_max = resume_budget if reason == 'drop' else base_max
        if rounds >= max_rounds or next_prompt_est + (next_max or 0) > context_budget:
            if verbose or reason == 'drop':
                print(f'{label}: stopping after round {rounds} (max_rounds/context_budget guard); '
                      f'output still {"dropped" if reason == "drop" else "truncated"}.')
            break

        kwargs['max_tokens'] = next_max
        if raw is not None:
            raw.trim()
        else:
            carry = '\n'.join(p for p in partials if p) if continue_carry == 'all' else partials[-1]
            template = continue_template or (DEFAULT_CONTINUE_TEMPLATE if reason == 'length'
                                             else DEFAULT_DROP_CONTINUE_TEMPLATE)
            kwargs['messages'] = _continue_messages(continue_mode, messages, carry, template)
        if reason == 'drop':
            await asyncio.sleep(drop_resume_wait)  # drops come in bursts; let the server recover
        elif verbose:
            print(f'auto_continue: round {rounds} truncated, re-querying ({continue_mode}) with '
                  f'~{next_prompt_est} prompt tokens...')

    if raw is not None and rounds > 1:
        full_reasoning, full_text = raw.split()
    else:
        full_text = '\n'.join(p for p in text_parts if p) or (text_parts[-1] if text_parts else None)
        full_reasoning = '\n'.join(p for p in reasoning_parts if p) or None
    return {'text': full_text, 'reasoning_text': full_reasoning, **agg,
            'response': last_response, 'finish_reason': finish_reason, 'rounds': rounds,
            'drop_resumes': drop_resumes, 'usage_estimated': usage_estimated}

# ----------------------------------------------------------------------------- the local server
# A tiny model, unedited or edited, is scored through an OpenAI-compatible server on
# config.SERVER_PORT: `vllm serve`, or tools/serve_transformers.py for a model vLLM cannot load.
# The server runs on the Linux side (inside WSL on Windows, see config.shell_command) in the
# virtual environment config.VLLM_VENV, one model at a time. wsp_local_model registers what to
# serve, switch_local_server starts it, and local_query_async talks to it.

# Models whose server this process has already verified or launched (see local_query_async).
_local_server_verified = set()

_local_server_logs = {}  # model -> Linux-side path of the log of the last launch made by this process

# What can be served, filled at run time by wsp_local_model: key -> entry. An entry holds
#   'engine' ('vllm' or 'transformers'), 'port', 'hf_repo' (the model folder as the Linux side
#   sees it), 'server_args', 'launch_timeout' (seconds), 'max_model_len',
#   'thinking_mode' ('enable_thinking' or 'none'), 'prompt_mode' ('chat' or 'completion'),
#   the default sampling settings ('max_tokens', 'temperature', 'top_p', 'top_k',
#   'presence_penalty', 'repetition_penalty', 'stop_token_ids'),
#   'server_script' (engine 'transformers' only), and
#   'wsp': {'session_id', 'tiny_model', 'task', 'protocol', 'model_dir'}.
LOCAL_MODEL_CONFIG = {}

def is_server_running(host='localhost', port=None, timeout=2):
    """True if a server answers /health on `port` (default config.SERVER_PORT)."""
    port = config.SERVER_PORT if port is None else port
    try:
        r = urllib.request.urlopen(f'http://{host}:{port}/health', timeout=timeout)
        return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False

def wait_for_server(port=None, timeout=500, interval=2):
    """Block until the server on `port` accepts requests (vLLM's /health only answers 200 once the
    model is loaded and warmed up)."""
    port = config.SERVER_PORT if port is None else port
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running(port=port):
            return True
        time.sleep(interval)
    raise TimeoutError(f'server on port {port} did not become ready within {timeout}s')

def get_running_model(port=None):
    """The model name the server on `port` reports at /v1/models, or None."""
    port = config.SERVER_PORT if port is None else port
    try:
        r = requests.get(f'http://localhost:{port}/v1/models', timeout=3)
        if r.status_code == 200:
            data = r.json().get('data') or []
            return data[0]['id'] if data else None
    except requests.ConnectionError:
        pass
    return None

def _local_launch_command(model):
    entry = LOCAL_MODEL_CONFIG[model]
    activate = f'source {config.VLLM_ACTIVATE}'  # not quoted, so that bash expands a leading ~
    model_dir = shlex.quote(entry['hf_repo'])
    if entry['engine'] == 'vllm':
        return (f"{activate} && {config.SERVER_ENV}vllm serve {model_dir} "
                f"--served-model-name {model} --enable-prefix-caching --gpu-memory-utilization 0.85 "
                f"{entry['server_args']} --port {entry['port']}")
    if entry['engine'] == 'transformers':
        # A tiny model that vLLM cannot load: the benchmark's own small completions server, in the
        # vLLM environment because that one has torch and transformers.
        return (f"{activate} && python -u {shlex.quote(entry['server_script'])} --model-dir {model_dir} "
                f"--served-model-name {model} {entry['server_args']} --port {entry['port']}")
    raise ValueError(f"unknown engine {entry['engine']!r} for model {model!r}")

def _local_launch_shell(model):
    """(bash command line, Linux-side log path) for launching `model` with its console output also
    written to a timestamped file under config.SERVER_LOG_DIR. The server's console closes when
    the server dies, so without the file a crash leaves nothing to read."""
    log_dir = config.SERVER_LOG_DIR
    log_path = f"{log_dir}/{model}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    cmd = _local_launch_command(model)
    return f'mkdir -p {log_dir} && ( {cmd} ) 2>&1 | tee {log_path}', log_path

def ensure_local_server(model, auto_launch=True, verbose=True):
    """Make sure `model` is being served; launch it if nothing is on its port. On Windows the
    server gets its own console window; elsewhere it runs detached in the background.
    Raises if the port is busy with a different model (stop it with stop_local_server first).
    Call this once before a run, so that the launch happens outside the event loop.
    A launch's console output is also written to a file under config.SERVER_LOG_DIR (printed at
    launch; read it with local_server_log(model))."""
    entry = LOCAL_MODEL_CONFIG[model]
    port = entry['port']
    if is_server_running(port=port):
        running = get_running_model(port)
        if running != model:
            raise ValueError(f"Port {port} is serving {running!r}, not {model!r}. "
                             f"Run stop_local_server() first.")
        return
    if not auto_launch:
        raise RuntimeError(f"No {entry['engine']} server on port {port} and auto_launch=False.")
    cmd, log_path = _local_launch_shell(model)
    _local_server_logs[model] = log_path
    if verbose: print(f"[local] launching {entry['engine']} (console log: {log_path}): {cmd}")
    if config.IS_WINDOWS:
        subprocess.Popen(config.shell_command(cmd), creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(config.shell_command(cmd), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_server(port=port, timeout=entry.get('launch_timeout', 120))

def local_server_log(model=None, tail=60, grep=None, verbose=True):
    """The last `tail` lines of the most recent server log for `model` (any model if None), read
    from config.SERVER_LOG_DIR on the Linux side. `grep` keeps only the lines containing that text
    (e.g. 'Error', 'Traceback', 'Killed'). Returns the lines; prints them when verbose. Use it
    after a run whose local requests failed with connection errors."""
    log_dir = config.SERVER_LOG_DIR
    pattern = f"{log_dir}/{model}_*.log" if model else f"{log_dir}/*.log"
    script = (f"f=$(ls -t {pattern} 2>/dev/null | head -1); "
              f"if [ -z \"$f\" ]; then echo '[no log matches {pattern}]'; else echo \"== $f\"; tail -n {int(tail)} \"$f\"; fi")
    out = subprocess.run(config.shell_command(script), capture_output=True, text=True, encoding='utf-8',
                         errors='replace').stdout
    lines = [l for l in out.splitlines() if grep is None or grep in l or l.startswith('==')]
    if verbose:
        for l in lines: print(l)
    return lines

def stop_local_server(engine=None, port=None, wait=60):
    """Kill the running server process (vLLM and the transformers server alike) and wait for the
    GPU to free. Polls the port until it stops answering, up to `wait` seconds, then gives the
    driver a few more seconds to release GPU memory. `engine` is accepted for compatibility and
    ignored: both engines share one port, so stopping either stops both."""
    port = config.SERVER_PORT if port is None else port
    for pattern in ('vllm serve', 'serve_transformers.py'):
        subprocess.run(config.shell_command(f'pkill -f {shlex.quote(pattern)}'))
    _local_server_verified.clear()  # the next local_query_async call re-checks (and may relaunch)
    # Drop the cached client for this port: its keep-alive connections point at the dead server and
    # the first request after a relaunch would fail with APIConnectionError.
    for base_url in [u for u in _local_clients if f':{port}/' in u]:
        _local_clients.pop(base_url, None)
    t0 = time.time()
    while is_server_running(port=port) and time.time() - t0 < wait:
        time.sleep(1)
    time.sleep(5)

def switch_local_server(model, verbose=True):
    """ensure_local_server(model), first stopping whatever other model occupies the port. Call it
    once before a run, outside the event loop."""
    entry = LOCAL_MODEL_CONFIG[model]
    port = entry['port']
    if is_server_running(port=port):
        running = get_running_model(port)
        if running != model:
            if verbose: print(f'[local] port {port} is serving {running!r}; stopping it to launch {model!r}')
            stop_local_server(port=port)
    ensure_local_server(model, auto_launch=True, verbose=verbose)
    _local_server_verified.add(model)

def _local_thinking(entry, thinking):
    """extra_body carrying the thinking switch of a model whose chat template has one."""
    if entry['thinking_mode'] == 'enable_thinking':
        # With False the template inserts an empty think block, which suppresses reasoning.
        return {'chat_template_kwargs': {'enable_thinking': bool(thinking)}}
    return {}

def _get_local_async_client(entry):
    base_url = f"http://localhost:{entry['port']}/v1"
    if base_url not in _local_clients:
        _local_clients[base_url] = AsyncOpenAI(api_key='not-needed', base_url=base_url,
                                               max_retries=0, timeout=LOCAL_REQUEST_TIMEOUT)
    return _local_clients[base_url]

def wsp_local_model(tiny_model=None, session_id=None, max_tokens=None):
    """Register the LOCAL_MODEL_CONFIG entry that serves a tiny model, and return its key.

    tiny_model : a key of config.TINY_MODEL_REGISTRY. With session_id=None the entry serves the
        UNEDITED model from the registry's model_dir; its key is 'wsp-<tiny_model>-base'.
    session_id : the ID of an edit session (a row of Master_WSP_Edits.csv) whose checkpoint was
        built. The entry serves the EDITED checkpoint <config.CHECKPOINT_DIR>/<ID>; its key is
        'wsp-<tiny_model>-<ID>'. The checkpoint's edits.json names its tiny model, so `tiny_model`
        is only needed for a checkpoint that does not.
    max_tokens : the default new-token limit of the entry; None -> the protocol's own limit.

    Everything else comes from config, so a base model and its edited checkpoints are served and
    sampled identically: the model's protocol for its task (sampling settings, chat or completion
    format), its server arguments, its --max-model-len and the stop ids named in its files. The
    key is the row's `model` in the score files (provider 'local') and the entry's 'wsp' dict says
    which session, tiny model, task and protocol it is. Calling it again for the same model just
    rebuilds the same entry. All entries use one port: serve one at a time, switching with
    switch_local_server(key)."""
    record = None
    if session_id is None:
        if tiny_model is None:
            raise ValueError('give `tiny_model` (the unedited model), `session_id` (an edited checkpoint), or both')
        model_dir = config.tiny_entry(tiny_model)['model_dir']
    else:
        model_dir = config.checkpoint_dir(session_id)
        record = config.read_checkpoint_record(model_dir)
        if record is None:
            raise FileNotFoundError(f'{model_dir} holds no edits.json: no checkpoint was built for session {session_id}')
        recorded = record.get('tiny_model')
        if recorded and tiny_model and recorded != tiny_model:
            raise ValueError(f'session {session_id} edited {recorded}, not {tiny_model}')
        tiny_model = recorded or tiny_model
        if tiny_model is None:
            raise ValueError(f'the checkpoint of session {session_id} does not name its tiny model; '
                             f'pass `tiny_model`')
    entry = config.tiny_entry(tiny_model)
    task, protocol_name = config.resolve_task(tiny_model, record.get('task') if record else None)
    protocol = config.PROTOCOLS[protocol_name]
    sampling, engine = protocol['sampling'], entry.get('engine', 'vllm')
    key = f'wsp-{tiny_model}-' + ('base' if session_id is None else str(session_id))
    LOCAL_MODEL_CONFIG[key] = {
        'engine': engine, 'port': config.SERVER_PORT, 'thinking_mode': entry['thinking_mode'],
        'hf_repo': config.to_wsl_path(model_dir),  # a local folder, served as plain files
        'max_tokens': sampling['max_tokens'] if max_tokens is None else max_tokens,
        'temperature': sampling['temperature'], 'top_p': sampling.get('top_p'), 'top_k': sampling.get('top_k'),
        'presence_penalty': sampling.get('presence_penalty'), 'repetition_penalty': sampling.get('repetition_penalty'),
        'max_model_len': entry['max_model_len'],
        'stop_token_ids': config.stop_token_ids(model_dir, entry),
        'prompt_mode': protocol['format'],  # 'chat', or 'completion' for a model with no chat template
        'server_args': entry['serve_args'], 'launch_timeout': 900,
        'wsp': {'session_id': session_id, 'tiny_model': tiny_model, 'task': task, 'protocol': protocol_name,
                'model_dir': model_dir},
    }
    if engine == 'transformers':
        LOCAL_MODEL_CONFIG[key]['server_script'] = config.to_wsl_path(
            os.path.join(config.TOOLS_DIR, 'serve_transformers.py'))
    return key


# A request to the local server that could not connect, or whose connection died mid-body: the
# SDK wraps connect failures as APIConnectionError (APITimeoutError is a subclass); a stream that
# dies surfaces as the raw httpx error (STREAM_DROP_ERRORS) when resume_on_drop does not take it.
_LOCAL_CONNECTION_ERRORS = (APIConnectionError, httpx.ConnectError) + STREAM_DROP_ERRORS

def _note_local_connection_error(model, port, exc):
    """Called by local_query_async when a request to the local server fails at the connection
    level. The server has most likely gone away (a crash, a closed console, a WSL shutdown), so
    forget that it was verified: the next call re-runs ensure_local_server, which relaunches it,
    instead of every remaining question of the run failing in seconds. Also write the traceback
    to the error log once per outage, since ut.report_exc keeps APIConnectionError out of it. One
    print per outage; the callers' retry ladders print the per-question lines as usual."""
    if model not in _local_server_verified:
        return  # already noted by another worker of this outage; the next call will re-check
    _local_server_verified.discard(model)
    path = ut.log_exc(exc, f'local server {model} on port {port}: connection error; '
                           f'the next local_query_async call re-checks the port and relaunches')
    log = _local_server_logs.get(model)
    print(f'[local] {model} on port {port}: {ut.describe_exc(exc)}. The server will be re-checked and '
          f'relaunched on the next call'
          + (f' (full trace: {path})' if path else '')
          + (f'; its console log is {log}' if log else f'; its console log is under {config.SERVER_LOG_DIR}'),
          flush=True)

async def _local_completion_query(entry, model, prompt, system_prompt, max_tokens, temperature, top_p, extra_body,
                                  server_provider, tools=None, verbose=True):
    """local_query_async for an entry whose 'prompt_mode' is 'completion' (a model with no chat
    template, such as TinyStories-33M): POST /v1/completions with the prompt exactly as given, one
    round. Returns the response contract; 'text' is the raw continuation, not stripped (a leading
    space is part of the answer).

    What a completion model cannot take is refused instead of dropped silently: a system prompt
    and tools. What does not apply is ignored: `thinking`, and the continuation arguments. The
    reply arrives in one piece, since the transformers server does not stream; with verbose it is
    printed like a chat reply once it is in."""
    if system_prompt is not None:
        raise ValueError(f'{model} is served in completion mode (it has no chat template), so it takes no system prompt')
    if tools is not None:
        raise ValueError(f'{model} is served in completion mode (it has no chat template), so it takes no tools')
    kwargs = {'model': model, 'prompt': prompt,
              'max_tokens': entry['max_tokens'] if max_tokens is None else max_tokens,
              'temperature': entry['temperature'] if temperature is None else temperature}
    top_p = entry.get('top_p') if top_p is None else top_p
    if top_p is not None: kwargs['top_p'] = top_p
    if extra_body: kwargs['extra_body'] = extra_body
    try:
        response = await _get_local_async_client(entry).completions.create(**kwargs)
    except _LOCAL_CONNECTION_ERRORS as exc:
        _note_local_connection_error(model, entry.get('port'), exc)
        raise
    choice, usage = response.choices[0], response.usage
    truncated = choice.finish_reason == 'length'
    ut.live_status_tick(len(choice.text or ''))  # the live footer counts it as it would count stream deltas
    if verbose:
        # the same shape as a chat reply (_chat_query_loop), with the prompt shown first because
        # the text continues it
        print(f'{prompt}{choice.text}')
        print(f"\nTokens -- prompt: {usage.prompt_tokens} (cached: 0), reasoning: 0, "
              f"completion: {usage.completion_tokens}, total: {usage.total_tokens}"
              + (f" [cut at max_tokens={kwargs['max_tokens']}]" if truncated else ''))
    return {
        'text': choice.text, 'reasoning_text': None,
        'prompt_tokens': usage.prompt_tokens, 'cached_tokens': 0, 'cache_write_tokens': 0,
        'completion_tokens': usage.completion_tokens, 'reasoning_tokens': 0, 'total_tokens': usage.total_tokens,
        'cost_estimate': 0.0, 'response': response,
        'status': 'incomplete' if truncated else 'completed',
        'incomplete_reason': 'max_tokens' if truncated else None,
        'server_provider': server_provider, 'engine': entry['engine'], 'rounds': 1, 'drop_resumes': 0,
        'usage_estimated': False,
    }

def split_inline_thinking(text, think_tags=THINK_TAGS):
    """(answer text, reasoning text) of a reply whose think block is written inside the text.

    The answer is what follows the LAST closing tag, as in tools/run_practice_baseline.py; the
    reasoning is everything before it, without the opening tag. A reply that opens a think block
    and never closes it (cut off while thinking) is all reasoning, with '' as its answer, which is
    what a server with a reasoning parser returns for it. Text with no think tags comes back
    unchanged, with None as its reasoning."""
    if not text:
        return text, None
    open_tag, close_tag = think_tags
    stripped = text.strip()
    if close_tag in stripped:
        reasoning, _, answer = stripped.rpartition(close_tag)
    elif stripped.startswith(open_tag):
        reasoning, answer = stripped, ''
    else:
        return text, None
    if reasoning.lstrip().startswith(open_tag):
        reasoning = reasoning.lstrip()[len(open_tag):]
    return answer.strip(), reasoning.strip() or None

async def local_query_async(prompt, model, thinking=True, system_prompt=None,
                            max_tokens=None, temperature=None, top_p=None, top_k=None,
                            tools=None, stream=False, verbose=True, auto_launch=True,
                            auto_continue=False, max_rounds=8, context_budget=None,
                            continue_template=None, continue_include_reasoning=True,
                            continue_carry='all', continue_mode='default', resume_on_drop=False,
                            max_drop_resumes=2, drop_resume_wait=5):
    """Query a tiny model on the local server.

    model : a key of LOCAL_MODEL_CONFIG (as returned by wsp_local_model), or a key of
        config.TINY_MODEL_REGISTRY ('gemma-3-270m-it', 'TinyStories-33M', ...): the unedited
        model's entry is then registered through wsp_local_model.
    thinking : for a model whose chat template has a thinking switch; None means True.
    system_prompt, max_tokens, temperature, top_p, top_k : override the entry's settings (the
        model's protocol) for this call.
    auto_launch : launch the server if nothing is on its port.
    auto_continue, max_rounds, context_budget, continue_*, resume_on_drop, max_drop_resumes,
    drop_resume_wait : the continuation options of _chat_query_loop (off by default; see the
        comment above DEFAULT_CONTINUE_TEMPLATE). context_budget defaults to the entry's
        'max_model_len', the server's --max-model-len.

    Returns the response contract: 'cost_estimate' is always 0 and 'server_provider' is
    'local/<engine>:<port>'; plus 'engine', 'rounds', 'drop_resumes' and 'usage_estimated'. Only
    an interrupted stream returns the reduced contract.

    When the server returns no separate reasoning and the text carries a think block (a model
    served without a reasoning parser), the block is moved to 'reasoning_text' and 'text' is what
    follows it (split_inline_thinking). A reply cut off while thinking therefore has text '', as
    with a parser. 'reasoning_tokens' stays as the server reported it (0 for such a server).

    A model with no chat template (TinyStories-33M) is in completion mode
    (_local_completion_query): the prompt is continued as written, a system prompt or tools
    raise, and 'text' is not stripped."""
    if model not in LOCAL_MODEL_CONFIG:
        if model not in config.TINY_MODEL_REGISTRY:
            raise ValueError(f'`model` must be one of {list(LOCAL_MODEL_CONFIG)}, or a tiny model: '
                             f'{list(config.TINY_MODEL_REGISTRY)}')
        model = wsp_local_model(tiny_model=model, session_id=None, max_tokens=max_tokens)
    entry = LOCAL_MODEL_CONFIG[model]
    engine, port = entry['engine'], entry['port']
    thinking = True if thinking is None else thinking
    server_provider = f'local/{engine}:{port}'

    if model not in _local_server_verified:
        # The server is checked once per process, not per call: ensure_local_server makes blocking
        # HTTP round trips, and doing that inside every call stalls a run with many concurrent
        # workers. stop_local_server() and a connection error (_note_local_connection_error) clear
        # _local_server_verified so that the next call re-checks and can relaunch.
        ensure_local_server(model, auto_launch=auto_launch, verbose=verbose)
        _local_server_verified.add(model)

    extra_body = _local_thinking(entry, thinking)
    top_k = entry['top_k'] if top_k is None else top_k
    if top_k is not None:
        extra_body['top_k'] = top_k
    if entry.get('repetition_penalty') is not None:
        extra_body['repetition_penalty'] = entry['repetition_penalty']
    if entry.get('stop_token_ids'):  # the end-of-sequence ids the model's own files name
        extra_body['stop_token_ids'] = entry['stop_token_ids']
    if entry.get('prompt_mode') == 'completion':
        return await _local_completion_query(entry, model, prompt, system_prompt, max_tokens, temperature, top_p,
                                             extra_body, server_provider, tools=tools, verbose=verbose)

    messages = []
    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    kwargs = {
        'model': model,
        'messages': messages,
        'stream': stream,
        'max_tokens': entry['max_tokens'] if max_tokens is None else max_tokens,
        'temperature': entry['temperature'] if temperature is None else temperature,
        'top_p': entry['top_p'] if top_p is None else top_p,
        'presence_penalty': entry.get('presence_penalty')
    }
    if extra_body: kwargs['extra_body'] = extra_body
    if tools is not None: kwargs['tools'] = tools
    if stream:
        kwargs['stream_options'] = {'include_usage': True}  # needed for the usage chunk

    if context_budget is None:
        context_budget = entry.get('max_model_len', 240_000)
    client = _get_local_async_client(entry)

    try:
        loop = await _chat_query_loop(
            client, kwargs, messages, stream=stream, verbose=verbose, label=f'{model} ({server_provider})',
            local_timeout=None, rates=None, auto_continue=auto_continue, max_rounds=max_rounds,
            context_budget=context_budget, continue_template=continue_template,
            continue_include_reasoning=continue_include_reasoning, continue_carry=continue_carry,
            continue_mode=continue_mode, resume_on_drop=resume_on_drop,
            max_drop_resumes=max_drop_resumes, drop_resume_wait=drop_resume_wait,
            raw_source=(entry.get('hf_repo'), None, THINK_TAGS),
            context_window=entry.get('max_model_len'))
    except _LOCAL_CONNECTION_ERRORS as exc:
        _note_local_connection_error(model, port, exc)
        raise
    if loop.pop('interrupted', False):
        return loop  # interrupted stream: reduced contract, prior rounds kept

    finish_reason = loop['finish_reason']
    status = 'incomplete' if finish_reason in ('length', 'dropped') else 'completed'
    incomplete_reason = {'length': 'max_tokens', 'dropped': 'stream_dropped'}.get(finish_reason)
    full_text = loop['text'].strip() if loop['text'] else loop['text']
    reasoning_text = loop['reasoning_text']
    if not reasoning_text:
        # A server with no reasoning parser returns the think block inside the text. Split it
        # out, so that the contract is the same as with a parser.
        full_text, reasoning_text = split_inline_thinking(full_text)

    return {
        'text': full_text,
        'reasoning_text': reasoning_text,
        'prompt_tokens': loop['prompt_tokens'],
        'cached_tokens': loop['cached_tokens'],
        'cache_write_tokens': 0,
        'completion_tokens': loop['completion_tokens'],
        'reasoning_tokens': loop['reasoning_tokens'],
        'total_tokens': loop['total_tokens'],
        'cost_estimate': 0.0,
        'response': loop['response'],
        'status': status,
        'incomplete_reason': incomplete_reason,
        'server_provider': server_provider,
        'engine': engine,
        'rounds': loop['rounds'],
        'drop_resumes': loop['drop_resumes'],
        'usage_estimated': loop['usage_estimated'],
    }
