"""Run edit sessions: a primary LLM is shown a tiny model and replies once with an edit to its weights.

Where it sits in the flow: the top. Each session calls the primary LLM's API, which costs real
money (docs and the paper give the cost per model), builds the edited checkpoint under
cfg.CHECKPOINT_DIR and appends a row to <results>/Master_WSP_Edits.csv. Scoring the checkpoint is
a separate step (scripts/score_tiny_model.py). Run from the repository root:

    python scripts/run_sessions.py --model gpt-6-astra --tiny-model gemma-3-270m-it --tier oneshot_spec

For a primary LLM of the paper, --model alone selects its provider and the settings the paper's
sessions used (PRESETS below); --effort changes the reasoning effort. Any other model registered in
wspmathbench/llm/registry.py needs --provider, and takes --effort and --llm-kwargs.

A trial that costs nothing and needs no API key uses the stub provider, whose canned replies
exercise the whole path (models: stub-spec, stub-spec-invalid-then-valid, stub-code,
stub-code-raises, stub-no-block). With --smoketest it writes into <results>/smoketest/ and adds no
row:

    python scripts/run_sessions.py --provider stub --model stub-spec --smoketest

The oneshot_code tier runs the primary LLM's program on this machine with the user's own
permissions (wspmathbench/kernel.py says what limits it and what does not).

From Python, run_sessions(n_runs, max_concurrent, **arguments) passes every argument of
wspmathbench.benchmark.WSP_Benchmark through.
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root

from wspmathbench import benchmark  # noqa: E402
from wspmathbench import config as cfg  # noqa: E402
from wspmathbench import session  # noqa: E402

# A oneshot_code session runs a CPU-bound program for up to cfg.SANDBOX['timeout_s'] seconds with up
# to cfg.SANDBOX['threads'] threads, so few sessions run at once.
DEFAULT_MAX_CONCURRENT = 2

# The provider and the llm_kwargs of the paper's sessions, per primary LLM. `reasoning_effort` is the
# highest setting each model offered at the time, except claude-fable-5-2 (medium) and
# gemini-4-carbon (high). 'host' None is the provider's own API; 'OpenRouter' routes through
# OpenRouter. 'flex' is OpenAI's lower-priced service tier and does not change the request otherwise.
PRESETS = {
    'gpt-6.1-sol': ('OpenAI', {'reasoning_effort': 'max', 'host': None, 'stream': True, 'flex': True}),
    'gpt-6-astra': ('OpenAI', {'reasoning_effort': 'max', 'host': None, 'stream': True, 'flex': True}),
    'gpt-6-astra-pro': ('OpenAI', {'reasoning_effort': 'max', 'host': None, 'stream': True, 'flex': True}),
    'claude-fable-5-2': ('Anthropic', {'reasoning_effort': 'medium', 'host': None, 'stream': True,
                                       'max_tokens': 128000}),
    # Three of the paper's ten gemini-4-carbon sessions ran with max_tokens 128000, two of them on
    # Google's own API (host None); the other seven ran with the settings below.
    'gemini-4-carbon': ('Google', {'reasoning_effort': 'high', 'host': 'OpenRouter', 'stream': True,
                                   'max_tokens': 300000}),
    'deepseek-v4.1-flash-1017': ('DeepSeek', {'reasoning_effort': 'high', 'host': 'OpenRouter', 'stream': True,
                                              'thinking': True}),
    'kimi-k3.1-pro': ('Kimi', {'reasoning_effort': 'max', 'host': 'OpenRouter', 'stream': True}),
}


def run_sessions(n_runs, max_concurrent=DEFAULT_MAX_CONCURRENT, **benchmark_arguments):
    """n_runs sessions of one configuration, at most max_concurrent at a time. Returns the IDs
    (None for each smoketest session)."""
    return asyncio.run(_run_sessions_async(n_runs, max_concurrent, benchmark_arguments))


async def _run_sessions_async(n_runs, max_concurrent, benchmark_arguments):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def one_session():
        async with semaphore:
            return await benchmark.WSP_Benchmark_async(**benchmark_arguments)

    return await asyncio.gather(*[one_session() for _ in range(n_runs)])


def resolve_model(model, provider=None, effort=None, extra_llm_kwargs=None):
    """(provider, llm_kwargs) for a primary LLM: its preset if it has one, with `effort` and
    `extra_llm_kwargs` laid over it; otherwise the given provider with streaming on."""
    if provider == session.STUB_PROVIDER:
        return provider, {}
    if model in PRESETS:
        preset_provider, llm_kwargs = PRESETS[model]
        provider, llm_kwargs = provider or preset_provider, dict(llm_kwargs)
    elif provider is None:
        raise ValueError(f'{model!r} has no preset (the presets are {list(PRESETS)}), so --provider is needed.')
    else:
        llm_kwargs = {'stream': True}
    if effort is not None:
        llm_kwargs['reasoning_effort'] = effort
    llm_kwargs.update(extra_llm_kwargs or {})
    return provider, llm_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tier', default='oneshot_spec', choices=cfg.TIERS)
    parser.add_argument('--tiny-model', default='Qwen3.5-0.8B', choices=list(cfg.TINY_MODEL_REGISTRY))
    parser.add_argument('--provider', default=None,
                        help="the model's provider in wspmathbench/llm/registry.py, or 'stub'; "
                             'not needed for a model with a preset')
    parser.add_argument('--model', required=True, help=f'the primary LLM; presets: {", ".join(PRESETS)}')
    parser.add_argument('--effort', default=None, help="reasoning effort, replacing the preset's")
    parser.add_argument('--llm-kwargs', default=None, type=json.loads,
                        help='a JSON object of further arguments for the provider\'s query function, '
                             'laid over the preset, for example \'{"max_tokens": 64000}\'')
    parser.add_argument('--n-runs', type=int, default=1, help='sessions of this configuration')
    parser.add_argument('--max-concurrent', type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument('--smoketest', action='store_true',
                        help='write into <results>/smoketest/ (cleared first) and add no master row; one session at a time')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='do not build the edited checkpoint (it is needed for scoring)')
    parser.add_argument('--notes', default=None, help="free text for the row's misc['Notes']")
    args = parser.parse_args()

    if args.smoketest and args.n_runs != 1:
        parser.error('a smoketest clears its folder first, so it runs one session: leave --n-runs at 1')
    try:
        provider, llm_kwargs = resolve_model(args.model, args.provider, args.effort, args.llm_kwargs)
    except ValueError as exc:
        parser.error(str(exc))
    ids = run_sessions(
        n_runs=args.n_runs, max_concurrent=args.max_concurrent,
        tier=args.tier, tiny_model=args.tiny_model, provider=provider, model=args.model, llm_kwargs=llm_kwargs,
        build_checkpoint=not args.no_checkpoint, smoketest=args.smoketest, Notes=args.notes,
        live_status=None if provider == session.STUB_PROVIDER else 'tokens')
    if not args.smoketest:
        print(f'session IDs: {ids}')


if __name__ == '__main__':
    main()
