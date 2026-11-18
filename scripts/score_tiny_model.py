"""Score a tiny model, unedited or edited, on its held-out evaluation sets.

Run it from the repository root. It starts the tiny model's server (stopping any other model on the
same port), asks every question of each evaluation set `--repeats` times, and appends one row per
pass to results/scores/Master_LLM_<evaluation set>_Results.csv. It needs the GPU; GSM8K-500 and
MATH-500 passes also need OPENAI_API_KEY, for the model that reads the final answer out of a reply.

The unedited model, once per tiny model:

    python scripts/score_tiny_model.py --tiny-model Qwen3.5-0.8B

The edited checkpoint of edit session 12:

    python scripts/score_tiny_model.py --session-id 12

Score an edited model and its unedited model with the same --max-tokens. See
wspmathbench/llm/scoring.py (score_tiny_model) for what each argument does.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wspmathbench import config as cfg
from wspmathbench.llm import benchmarks, scoring


def main():
    parser = argparse.ArgumentParser(description='Score a tiny model, unedited or edited, on its evaluation sets.')
    parser.add_argument('--tiny-model', choices=list(cfg.TINY_MODEL_REGISTRY),
                        help='score the UNEDITED model (leave --session-id out)')
    parser.add_argument('--session-id', type=int,
                        help='score the EDITED checkpoint of this edit session (a row of Master_WSP_Edits.csv)')
    parser.add_argument('--repeats', type=int, default=3, help='passes per evaluation set (default 3)')
    parser.add_argument('--max-tokens', type=int,
                        help="new tokens per answer (default: the protocol's own limit)")
    parser.add_argument('--max-workers', type=int,
                        help='questions in flight (default: 128, or 16 for TinyStories-33M)')
    parser.add_argument('--datasets', nargs='+', choices=list(benchmarks.EVAL_SETS),
                        help="evaluation sets (default: the tiny model's own)")
    parser.add_argument('--smoketest', action='store_true',
                        help='ask only the first question of each set and save nothing')
    parser.add_argument('--notes', help='free text added to the Notes of every row')
    args = parser.parse_args()
    if args.tiny_model is None and args.session_id is None:
        parser.error('give --tiny-model (the unedited model) or --session-id (an edited checkpoint)')

    scoring.score_tiny_model(
        tiny_model=args.tiny_model, session_id=args.session_id, datasets=args.datasets,
        repeats=args.repeats, max_tokens=args.max_tokens, max_workers=args.max_workers,
        notes=args.notes, smoketest=args.smoketest)


if __name__ == '__main__':
    main()
