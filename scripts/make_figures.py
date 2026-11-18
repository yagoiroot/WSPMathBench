"""Compile the WSPMathBench scores from the saved results and draw the figures.

Run from the repository root, after edit sessions have been run and the unedited and edited tiny
models have been scored:

    python scripts/make_figures.py                  # scores, score figures, bar figures
    python scripts/make_figures.py --no-bars        # scores and score figures only
    python scripts/make_figures.py --sessions-grid  # also the grid of which sessions exist

It prints the table of scores, writes it to <results>/compiled_wsp_benchmark_metrics.parquet, and
writes the figures, as PDF and PNG, to <results>/figures/:

    wsp_score.*, wsp_score_<tier>.*           the score of each primary LLM, both tiers and each alone
    bars_<tiny model>_<evaluation set>_<tier>.*   one tiny model on one evaluation set after the
                                              edits of each primary LLM, next to the unedited model
    completed_sessions.png                    with --sessions-grid

Only a primary LLM with a session for every tiny model in every tier, all of them scored, gets a
score; what is missing for the others is printed. Calls no API and needs no GPU. Grading MATH-500
answers starts a math_verify worker process, so the first run takes a few minutes.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')  # figures go to files; nothing is shown

from wspmathbench import analysis
from wspmathbench import config as cfg


def print_scores(df):
    """The compiled scores as a table: one row per primary LLM, one column per tier, in points."""
    tiers = list(dict.fromkeys(df['tier'].to_list()))
    entry = {(r['model'], r['tier']): f"{100 * r['score'] + 0.0:.1f} +/- {100 * r['score_se']:.2f}"
             for r in df.iter_rows(named=True)}
    models = list(dict.fromkeys(df['model'].to_list()))
    width = max(len(m) for m in models + ['primary LLM'])
    print()
    print(f'{analysis.BENCHMARK_NAME} score in points (+/- one standard error)')
    print('  '.join(['primary LLM'.ljust(width)] + [t.rjust(14) for t in tiers]))
    for m in models:
        print('  '.join([m.ljust(width)] + [entry.get((m, t), '-').rjust(14) for t in tiers]))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='family-wise confidence level of the tests (default 0.95, as in the paper)')
    parser.add_argument('--no-bars', action='store_true', help='skip the per-evaluation-set bar figures')
    parser.add_argument('--sessions-grid', action='store_true', help='also draw the grid of which sessions exist')
    args = parser.parse_args()

    if not os.path.exists(cfg.MASTER_CSV):
        sys.exit(f'There are no results yet: {cfg.MASTER_CSV} does not exist.\n'
                 f'Run edit sessions first (scripts/run_sessions.py), then score the unedited and the '
                 f'edited tiny models (scripts/score_tiny_model.py).')
    if args.sessions_grid:
        analysis.plot_completed_sessions(save=True, show=False)
        print(f"Saved completed_sessions.png to {cfg.FIGURES_DIR}")

    complete = analysis.complete_configurations()
    if not complete:
        sys.exit(f'No primary LLM has a session for every tiny model ({", ".join(cfg.TINY_MODEL_REGISTRY)}) in '
                 f'every tier ({", ".join(cfg.TIERS)}) yet, so no score can be computed.\n'
                 f'The sessions so far are listed in {cfg.MASTER_CSV}; --sessions-grid draws them.')
    if not os.path.isdir(cfg.SCORES_DIR):
        sys.exit(f'No tiny model has been scored yet: {cfg.SCORES_DIR} does not exist.\n'
                 f'Score the unedited and the edited tiny models first (scripts/score_tiny_model.py).')

    df = analysis.compile_scores(confidence=args.confidence)
    if df.height == 0:
        sys.exit('No score could be computed: see the list above of what has not been scored yet.')
    print_scores(df)

    tiers = list(dict.fromkeys(df['tier'].to_list()))
    for tier in (['both'] if len(tiers) > 1 else []) + tiers:
        analysis.plot_scores(tier=tier, save=True, show=False)

    if not args.no_bars:
        for tier in tiers:
            shown = df.filter(df['tier'] == tier)['model'].to_list()
            for tiny_model in cfg.TINY_MODEL_REGISTRY:
                for eval_set in analysis.eval_sets_of(tiny_model):
                    analysis.compare_scores(model=None, tiny_model=tiny_model, benchmark=eval_set, tier=tier,
                                            primary_llms=shown, confidence=args.confidence,
                                            save=True, show=False, verbose=False)
    print(f'Figures are in {cfg.FIGURES_DIR}')


if __name__ == '__main__':
    main()
