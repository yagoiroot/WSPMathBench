"""Analysis of saved WSPMathBench results: which sessions exist, how each edit changed a tiny
model's benchmark score, and the WSPMathBench score of each primary LLM.

    plot_completed_sessions()   a grid of which sessions exist: one row per primary LLM at a
                                reasoning effort, one column per tiny model, split by tier, with a
                                mark on the sessions whose edited model has been scored.
    select_sessions()           the session that counts for each (primary LLM, effort, tiny model,
                                tier): the latest one.
    compare_scores()            an unedited tiny model against its edited versions on one
                                evaluation set: mean scores, paired per-question tests with the
                                Holm-Bonferroni correction, and the bar figure.
    wsp_score()                 the WSPMathBench score of one primary LLM in one tier.
    compile_scores()            wsp_score for every primary LLM and tier, written to one file.
    plot_scores()               the bar figure of that file.

Reads Master_WSP_Edits.csv (one row per edit session) and results/scores/Master_LLM_<evaluation
set>_Results.csv (one row per scoring pass of a tiny model), both written by the benchmark itself,
and data/eval/<evaluation set>_Benchmark.jsonl for the reference answers. Writes only the compiled
score file and figures, under the results folder (see config.py).
"""
import atexit
import json
import math
import os
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from wspmathbench import config as cfg

BENCHMARK_NAME = 'WSPMathBench'


def scores_file(eval_set):
    """The results file of an evaluation set: one row per scoring pass of a tiny model."""
    return os.path.join(cfg.SCORES_DIR, f'Master_LLM_{eval_set}_Results.csv')


def scored_model_key(tiny_model, session_id=None):
    """The `model` value of a tiny model's rows in the results files: 'wsp-<tiny model>-base' for
    the unedited model, 'wsp-<tiny model>-<session ID>' for the model a session edited."""
    return f"wsp-{tiny_model}-{'base' if session_id is None else session_id}"


# ============================================================================================
# Which sessions exist
# ============================================================================================

# A session's status, best first. A cell takes the colour of the best status among its sessions.
# 'program_raised' is not a status of the master table: it is an 'empty_edit' row whose misc
# records an exception, that is, a oneshot_code program that raised before saving anything. A
# plain 'empty_edit' is a reply that was applied or run without error and changed nothing.
PROGRAM_RAISED = 'program_raised'
STATUS_ORDER = ('edit_saved', 'empty_edit', PROGRAM_RAISED, 'no_valid_reply', 'failed')
STATUS_COLOURS = {'edit_saved': '#5BA97B', 'empty_edit': '#E3B74A', PROGRAM_RAISED: '#E08A4F',
                  'no_valid_reply': '#CF6B55', 'failed': '#CF6B55'}
STATUS_LABELS = {'edit_saved': 'edit saved', 'empty_edit': 'ran without error, nothing edited',
                 PROGRAM_RAISED: 'program raised, nothing edited',
                 'no_valid_reply': 'no valid reply / failed', 'failed': 'no valid reply / failed'}
NOT_RUN_COLOUR = '#ECECEC'
OTHER_STATUS_COLOUR = '#CF6B55'


def _status_rank(status):
    return STATUS_ORDER.index(status) if status in STATUS_ORDER else len(STATUS_ORDER)


def completed_sessions(master_csv=None):
    """The sessions of the master table, grouped for the grid.

    Returns (rows, tiny_models, cells):
      rows         (primary model, reasoning effort) pairs, in order of first session
      tiny_models  the registry's tiny models, then any other that has a session
      cells        {(model, effort, tiny model, tier): [(session ID, status, task), ...]}, by ID. The
                   status is the master table's, except that an 'empty_edit' whose misc records
                   an exception becomes 'program_raised'
    The effort is llm_kwargs['reasoning_effort'], or 'default' when it is not set, as in the
    session folder names.
    """
    master_csv = master_csv or cfg.MASTER_CSV
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f'{master_csv} does not exist: no edit session has been run yet.')
    df = pl.read_csv(master_csv, infer_schema_length=None).sort('ID')
    rows, cells = [], {}
    tiny_models = list(cfg.TINY_MODEL_REGISTRY)
    for r in df.iter_rows(named=True):
        effort = str(json.loads(r['llm_kwargs'] or '{}').get('reasoning_effort') or 'default')
        if (r['model'], effort) not in rows:
            rows.append((r['model'], effort))
        if r['tiny_model'] not in tiny_models:
            tiny_models.append(r['tiny_model'])
        status = r['status']
        if status == 'empty_edit' and json.loads(r['misc'] or '{}').get('exception'):
            status = PROGRAM_RAISED
        cells.setdefault((r['model'], effort, r['tiny_model'], r['tier']), []).append((r['ID'], status, r['task']))
    return rows, tiny_models, cells


def benchmark_entries(tiny_model, session_id, task):
    """How many rows the edited model of a session has in each results file of its task.
    session_id='base' counts the rows of the unedited tiny model instead ('wsp-<tiny_model>-base').

    Returns {evaluation set: number of rows}, for the task's eval_sets in order (GSM8K_500 and
    MATH500 for 'math', BasicArithmetic for 'basic_arithmetic'). The rows are those of the set's
    results file (scores_file) whose model is 'wsp-<tiny_model>-<session ID>', the key under which
    an edited model is scored. A missing results file counts as 0.
    """
    key = scored_model_key(tiny_model, None if session_id == 'base' else session_id)
    return {eval_set: _results_models(eval_set).count(key) for eval_set in cfg.TASKS[task]['eval_sets']}


_results_models_cache = {}


def _results_models(eval_set):
    """The model column of an evaluation set's results file, read once per call of the plot."""
    if eval_set not in _results_models_cache:
        path = scores_file(eval_set)
        _results_models_cache[eval_set] = (pl.read_csv(path, columns=['model'], infer_schema_length=None)['model'].to_list()
                                           if os.path.exists(path) else [])
    return _results_models_cache[eval_set]


def _benchmark_mark(tiny_model, session_id, task, min_entries):
    """' ✓' when every results file of the task has at least min_entries rows for the session's
    edited model, ' (a/b)' with the counts when some rows exist but not enough, '' when none do.
    session_id='base' gives the mark of the unedited tiny model."""
    counts = list(benchmark_entries(tiny_model, session_id, task).values())
    if counts and min(counts) >= min_entries:
        return r' $\checkmark$'
    return f" ({'/'.join(str(n) for n in counts)})" if any(counts) else ''


def _glass_tile(ax, x, y, w, h, colour, flat=False):
    """Rounded tile with its lower left corner at (x, y), on an axes whose units are inches: a
    vertical fade of the colour with a coloured rim and a white inner highlight. flat=True
    gives a plain fill, for the cells that were not run."""
    style = 'round,pad=0,rounding_size=0.035'
    if flat:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=colour, edgecolor='none'))
        return
    c = np.array(to_rgb(colour))
    clip = FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor='none', edgecolor='none')
    ax.add_patch(clip)
    t = np.linspace(0, 1, 64)[:, None, None]
    body = np.ones((64, 1, 4))
    body[..., :3] = (1 - t) * (c + (1 - c) * 0.58) + t * (c + (1 - c) * 0.22)
    ax.imshow(body, extent=(x, x + w, y, y + h), origin='lower', aspect='auto',
              interpolation='bicubic', zorder=2).set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor='none',
                                edgecolor=(1, 1, 1, 0.55), linewidth=1.4, zorder=2.2)).set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor='none',
                                edgecolor=(*c, 0.75), linewidth=0.5, zorder=2.3))


# ~~~ plot_completed_sessions: shared pieces ~~~
# The figures are meant to stay legible when shown small, which they do only when they are small
# in inches for their font sizes: the three layouts stay near 5.2 x 5.0 in with text of 6 to
# 7.5 pt. Making a layout larger in inches makes its text smaller once the figure is scaled down.
_SESSIONS_DPI = 200
_INK, _INK_SOFT, _INK_FAINT, _WARN = '#1D1D24', '#55555F', '#8A8A96', '#B03A2E'
_GLASS_NOT_RUN_COLOUR = '#F1F1F4'


def _sessions_sans():
    """The sans-serif fonts of the session figures, best first, of those installed."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ('Inter', 'Segoe UI', 'Helvetica Neue', 'Arial') if f in installed] + ['DejaVu Sans']


def _wrap_at_hyphens(name, width):
    """A name broken at its hyphens into lines of at most `width` characters. A name that
    needs two lines is broken at the hyphen that leaves them closest in length."""
    lines, cur = [], ''
    for part in name.split('-'):
        trial = part if cur == '' else cur + '-' + part
        if len(trial) > width and cur != '':
            lines.append(cur + '-')
            cur = part
        else:
            cur = trial
    lines.append(cur)
    if len(lines) == 2:
        parts = name.split('-')
        splits = [('-'.join(parts[:k]) + '-', '-'.join(parts[k:])) for k in range(1, len(parts))]
        lines = list(min(splits, key=lambda pair: max(len(pair[0]), len(pair[1]))))
    return lines


def _session_cells(cells, min_entries):
    """What each cell that has sessions shows: ({cell key: (best status, [a label per session])},
    the set of best statuses). A label is the session ID, '*' when its status is worse than the
    cell's, then its benchmark mark."""
    info = {}
    for key, sessions in cells.items():
        best = min((status for _, status, _ in sessions), key=_status_rank)
        info[key] = (best, [f'{sid}' + ('' if status == best else '*') + _benchmark_mark(key[2], sid, task, min_entries)
                            for sid, status, task in sessions])
    return info, {best for best, _ in info.values()}


def _unedited_note(tiny, cells, min_entries):
    """(lines, benchmarked): the note under a tiny model's name on its unedited model, and
    whether that model has enough entries in every results file of its task."""
    if tiny in cfg.TINY_MODEL_REGISTRY:
        base_task = cfg.TINY_MODEL_REGISTRY[tiny]['default_task']
    else:
        base_task = next(s[0][2] for key, s in cells.items() if key[2] == tiny)
    mark = _benchmark_mark(tiny, 'base', base_task, min_entries)
    return (['unedited' + mark] if mark else ['unedited:', 'not benchmarked']), 'checkmark' in mark


def _key_colours(statuses_shown, not_run_colour):
    """[(colour, label)] of the statuses that appear in the figure, then 'not run'."""
    entries, seen = [], set()
    for status in STATUS_ORDER:
        if status in statuses_shown and STATUS_LABELS[status] not in seen:
            seen.add(STATUS_LABELS[status])
            entries.append((STATUS_COLOURS[status], STATUS_LABELS[status]))
    if statuses_shown - set(STATUS_ORDER):
        entries.append((OTHER_STATUS_COLOUR, 'other status'))
    return entries + [(not_run_colour, 'not run')]


def _key_marks(min_entries):
    """[(sample, meaning)] of the marks that follow a session ID."""
    return [(r'12 $\checkmark$', f'at least {min_entries} entries in each of its results files'),
            ('12 (3/1)', 'too few entries: the count in each file'),
            ('12*', "status worse than the tile's colour")]


def _sessions_key(w, colours, marks, stacked):
    """Lay out the key in a box `w` inches wide. stacked=True puts a 'Key' heading, the colours
    and the marks under one another, for a narrow slot; stacked=False puts the colours on the
    left and the marks on the right. Returns (height, items), where an item is
    ('tile', dx, dy, colour) or ('text', dx, dy, string, colour, size, weight), dx and dy being
    inches from the top left corner of the box (dy: the tile's top edge, the text's centre)."""
    size, line_h, sample_w = 6.3, 0.115, 0.45
    char_w = size * 0.47 / 72  # inches per character, on average

    def _wrap(s, width_in):
        return textwrap.wrap(s, max(8, int(width_in / char_w)))

    items = []
    if stacked:
        items.append(('text', 0, 0.07, 'Key', _INK, 7.5, 'bold'))
        y = 0.20
        for colour, label in colours:
            items.append(('tile', 0, y + 0.012, colour))
            for line in _wrap(label, w - 0.27):
                items.append(('text', 0.27, y + line_h / 2, line, _INK_SOFT, size, 'normal'))
                y += line_h
            y += 0.02
        y += 0.05
        for sample, meaning in marks:
            items.append(('text', 0, y + line_h / 2, sample, _INK, size, 'normal'))
            y += line_h
            for line in _wrap(meaning, w):
                items.append(('text', 0, y + line_h / 2, line, _INK_SOFT, size, 'normal'))
                y += line_h
            y += 0.03
        return y, items
    y = 0
    for colour, label in colours:
        items.append(('tile', 0, y + 0.012, colour))
        items.append(('text', 0.27, y + line_h / 2, label, _INK_SOFT, size, 'normal'))
        y += line_h + 0.01
    x2 = 0.27 + max(len(label) for _, label in colours) * char_w + 0.22
    y2 = 0
    for sample, meaning in marks:
        items.append(('text', x2, y2 + line_h / 2, sample, _INK, size, 'normal'))
        for line in _wrap(meaning, w - x2 - sample_w):
            items.append(('text', x2 + sample_w, y2 + line_h / 2, line, _INK_SOFT, size, 'normal'))
            y2 += line_h
        y2 += 0.01
    return max(y, y2), items


def _sessions_canvas(fig_w, fig_h, glass):
    """A figure of fig_w x fig_h inches with one axes that covers it, in inches. Returns
    (fig, ax, text, tile):
      text(x, top, s, size, colour, fit=None, **kw)  text centred `top` inches below the top of
          the figure; with fit, its font is shrunk until it is no wider than `fit` inches
      tile(x, top, w, h, colour, flat=False)  a tile whose top edge is `top` inches below the
          top of the figure: a glass tile, or a plain rectangle when glass is False or flat
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=_SESSIONS_DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    sans = _sessions_sans()

    def text(x, top, s, size=7, colour=_INK, fit=None, **kw):
        kw.setdefault('va', 'center')
        kw.setdefault('fontfamily', sans)  # named on the text, so a later draw or savefig keeps it
        t = ax.text(x, fig_h - top, s, fontsize=size, color=colour, zorder=4, **kw)
        if fit:
            width = t.get_window_extent().width / fig.dpi
            if width > fit:
                t.set_fontsize(max(size * fit / width, 4.5))
        return t

    def tile(x, top, w, h, colour, flat=False):
        if glass:
            _glass_tile(ax, x, fig_h - top - h, w, h, colour, flat=flat)
        else:
            ax.add_patch(Rectangle((x, fig_h - top - h), w, h, facecolor=colour, edgecolor='none'))

    return fig, ax, text, tile


def _draw_sessions_key(text, tile, x, top, items):
    """Draw the items of _sessions_key with the box's top left corner at (x, top)."""
    for item in items:
        if item[0] == 'tile':
            tile(x + item[1], top + item[2], 0.19, 0.10, item[3], flat=item[3] == _GLASS_NOT_RUN_COLOUR)
        else:
            _, dx, dy, s, colour, size, weight = item
            text(x + dx, top + dy, s, size=size, colour=colour, fontweight=weight)


def _sessions_titles(text, x, title, subtitle, fit):
    text(x, 0.14, title, size=9, fontweight='bold')
    text(x, 0.29, subtitle, size=6.3, colour=_INK_SOFT, fit=fit)


def _plot_completed_sessions_transposed(rows, tiny_models, cells, min_entries, save_path, save, show):
    """layout=3 of plot_completed_sessions: layout=2 turned around. A column per (primary LLM,
    effort), with the name of an LLM written once above its efforts; a block of rows per tiny
    model, one row per tier. The sessions of a cell are stacked, one per line, and a row is as
    tall as its fullest cell. The key runs along the bottom. Everything is placed in inches."""
    not_run_colour = _GLASS_NOT_RUN_COLOUR
    tiers = cfg.TIERS
    n_cols = len(rows)
    info, statuses_shown = _session_cells(cells, min_entries)
    # consecutive columns of one primary LLM share its name: [(model, first column, last column)]
    groups = []
    for i, (model, _) in enumerate(rows):
        if groups and groups[-1][0] == model:
            groups[-1][2] = i
        else:
            groups.append([model, i, i])
    # a name is written on one level line when its columns are wide enough for it; otherwise
    # it is turned on its side, on up to two lines
    cell_w = 0.335
    level = [len(model) * 0.052 <= (last - first + 1) * cell_w for model, first, last in groups]
    name_lines = [[model] if flat else _wrap_at_hyphens(model, 14) for (model, _, _), flat in zip(groups, level)]

    # ~~~ sizes, in inches ~~~
    m_side, m_top, m_bottom = 0.08, 0.40, 0.06
    label_w, tier_w = 0.82, 0.30           # tiny model name, then the tier of the row
    tile_gap, line_h = 0.02, 0.115
    name_h = 0.08 + 0.055 * max([len(line) for lines, flat in zip(name_lines, level) if not flat
                                 for line in lines] or [3])  # the LLM names on their side
    effort_h, block_gap, key_gap = 0.17, 0.07, 0.10
    row_h = {(tiny, tier): max(0.25, 0.06 + line_h * max(
                 [len(info[(m, e, tiny, tier)][1]) for m, e in rows if (m, e, tiny, tier) in info] or [1]))
             for tiny in tiny_models for tier in tiers}
    block_h = {tiny: sum(row_h[(tiny, tier)] for tier in tiers) for tiny in tiny_models}
    grid_x, grid_top = m_side + label_w + tier_w, m_top + name_h + effort_h
    grid_h = sum(block_h.values()) + (len(tiny_models) - 1) * block_gap
    fig_w = grid_x + n_cols * cell_w + m_side
    key_h, key_items = _sessions_key(fig_w - 2 * m_side, _key_colours(statuses_shown, not_run_colour),
                                     _key_marks(min_entries), stacked=False)
    fig_h = grid_top + grid_h + key_gap + key_h + m_bottom

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': _sessions_sans(),
                         'mathtext.fontset': 'dejavusans'}):
        fig, ax, _text, _tile = _sessions_canvas(fig_w, fig_h, glass=True)

        # ~~~ column heads: the LLM over its columns, then the effort of each column ~~~
        for (model, first, last), lines, flat in zip(groups, name_lines, level):
            xc = grid_x + (first + last + 1) / 2 * cell_w
            _text(xc, m_top + name_h - 0.05, '\n'.join(lines), size=7, rotation=0 if flat else 90,
                  ha='center', va='bottom', linespacing=1.05)
            ax.plot([grid_x + first * cell_w + 0.03, grid_x + (last + 1) * cell_w - 0.03],
                    [fig_h - m_top - name_h] * 2, color='#B9B9C4', linewidth=0.6)
        for i, (_, effort) in enumerate(rows):
            _text(grid_x + (i + 0.5) * cell_w, m_top + name_h + effort_h / 2, effort, size=6.3,
                  colour=_INK_SOFT, ha='center', fit=cell_w - 0.03)

        top0 = grid_top
        for tiny in tiny_models:
            # the tiny model's name, then the note on its unedited model, centred on the block
            note, benchmarked = _unedited_note(tiny, cells, min_entries)
            label = [(line, 7, _INK, 'bold') for line in _wrap_at_hyphens(tiny, 13)]
            label += [(line, 6.3, _INK_SOFT if benchmarked else _WARN, 'normal') for line in note]
            for n, (line, size, colour, weight) in enumerate(label):
                _text(m_side, top0 + block_h[tiny] / 2 + (n - (len(label) - 1) / 2) * 0.115, line,
                      size=size, colour=colour, fontweight=weight, fit=label_w - 0.04)
            top = top0
            for tier in tiers:
                rh = row_h[(tiny, tier)]
                _text(grid_x - 0.05, top + rh / 2, tier.replace('oneshot_', ''), size=6.3,
                      colour=_INK_SOFT, ha='right')
                for i, (model, effort) in enumerate(rows):
                    best, lines = info.get((model, effort, tiny, tier), (None, []))
                    colour = STATUS_COLOURS.get(best, OTHER_STATUS_COLOUR) if lines else not_run_colour
                    x = grid_x + i * cell_w
                    _tile(x + tile_gap / 2, top + tile_gap / 2, cell_w - tile_gap, rh - tile_gap, colour,
                          flat=not lines)
                    for n, line in enumerate(lines):
                        _text(x + cell_w / 2, top + rh / 2 + (n - (len(lines) - 1) / 2) * line_h, line,
                              size=7.5, ha='center', fit=cell_w - 0.05)
                top += rh
            top0 += block_h[tiny] + block_gap

        # ~~~ key along the bottom: the colours on the left, the marks on the right ~~~
        _draw_sessions_key(_text, _tile, m_side, grid_top + grid_h + key_gap, key_items)
        _sessions_titles(_text, m_side, f'{BENCHMARK_NAME} sessions run so far',
                         'Tiny model and edit tier, by primary LLM and reasoning effort. Numbers are session IDs.',
                         fig_w - 2 * m_side)
        ax.set_xlim(0, fig_w)  # imshow resets the limits
        ax.set_ylim(0, fig_h)

        if save:
            save_path = save_path or os.path.join(cfg.FIGURES_DIR, 'completed_sessions_transposed.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400)
        if show:
            plt.show()
    return fig


def _plot_completed_sessions_vertical(rows, tiny_models, cells, min_entries, save_path, save, show):
    """layout=2 of plot_completed_sessions: a panel per tiny model, three panels across, in the
    manner of the bar figures of compare_scores; in a panel, a row per (primary LLM, effort)
    and a column per tier. The rows are named once, to the left of the first panel of each
    row of panels. The key takes the slot after the last panel. Everything is placed in
    inches on one axes that covers the figure."""
    not_run_colour = _GLASS_NOT_RUN_COLOUR
    tiers = cfg.TIERS
    n_rows, n_tiers, panel_cols = len(rows), len(tiers), 3
    n_slots = len(tiny_models) + 1  # the panels, then the key
    panel_rows = math.ceil(n_slots / panel_cols)
    info, statuses_shown = _session_cells(cells, min_entries)

    # ~~~ sizes, in inches ~~~
    m_side, m_top, m_bottom = 0.08, 0.40, 0.06
    label_w, effort_w = 1.18, 0.40        # primary LLM name, then its effort
    cell_w, cell_h, tile_gap = 0.535, 0.142, 0.02
    head_h, panel_gap_x, panel_gap_y = 0.44, 0.10, 0.12
    panel_w, panel_h = n_tiers * cell_w, head_h + n_rows * cell_h
    key_h, key_items = _sessions_key(panel_w - 0.04, _key_colours(statuses_shown, not_run_colour),
                                     _key_marks(min_entries), stacked=True)
    # a row of slots that holds only the key is just as tall as the key; the row that holds
    # the key beside panels is at least as tall as the key
    key_row = len(tiny_models) // panel_cols
    row_h = [panel_h if r * panel_cols < len(tiny_models) else key_h for r in range(panel_rows)]
    row_h[key_row] = max(row_h[key_row], key_h)
    grid_x = m_side + label_w + effort_w
    fig_w = grid_x + panel_cols * panel_w + (panel_cols - 1) * panel_gap_x + m_side
    fig_h = m_top + sum(row_h) + (panel_rows - 1) * panel_gap_y + m_bottom

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': _sessions_sans(),
                         'mathtext.fontset': 'dejavusans'}):
        fig, ax, _text, _tile = _sessions_canvas(fig_w, fig_h, glass=True)

        def _slot(k):
            '''(left edge, distance of the top edge from the top of the figure) of slot k'''
            r, col = divmod(k, panel_cols)
            return (grid_x + col * (panel_w + panel_gap_x),
                    m_top + sum(row_h[:r]) + r * panel_gap_y)

        for j, tiny in enumerate(tiny_models):
            x0, top0 = _slot(j)
            note, benchmarked = _unedited_note(tiny, cells, min_entries)
            _text(x0 + 0.02, top0 + 0.075, tiny, size=7.5, fontweight='bold', fit=panel_w - 0.04)
            _text(x0 + 0.02, top0 + 0.195, ' '.join(note), size=6.3,
                  colour=_INK_SOFT if benchmarked else _WARN, fit=panel_w - 0.04)
            ax.plot([x0, x0 + panel_w], [fig_h - top0 - head_h + 0.15] * 2, color='#E6E6EB', linewidth=0.6)
            for k, tier in enumerate(tiers):
                _text(x0 + (k + 0.5) * cell_w, top0 + head_h - 0.07, tier.replace('oneshot_', ''),
                      size=6.3, colour=_INK_SOFT, ha='center')
            for i, (model, effort) in enumerate(rows):
                top = top0 + head_h + i * cell_h
                if j % panel_cols == 0:  # the rows are named once, to the left of the first panel
                    _text(grid_x - effort_w - 0.05, top + cell_h / 2, model, size=7, ha='right',
                          fit=label_w - 0.05)
                    _text(grid_x - effort_w, top + cell_h / 2, effort, size=6.3, colour=_INK_SOFT, ha='left')
                for k, tier in enumerate(tiers):
                    best, lines = info.get((model, effort, tiny, tier), (None, []))
                    colour = STATUS_COLOURS.get(best, OTHER_STATUS_COLOUR) if lines else not_run_colour
                    _tile(x0 + k * cell_w + tile_gap / 2, top + tile_gap / 2,
                          cell_w - tile_gap, cell_h - tile_gap, colour, flat=not lines)
                    _text(x0 + (k + 0.5) * cell_w, top + cell_h / 2, ', '.join(lines), size=7.5,
                          ha='center', fit=cell_w - 0.06)

        # ~~~ key, in the slot after the last panel ~~~
        x0, top = _slot(len(tiny_models))
        _draw_sessions_key(_text, _tile, x0 + 0.02, top, key_items)
        _sessions_titles(_text, m_side, f'{BENCHMARK_NAME} sessions run so far',
                         'Primary LLM and reasoning effort, by tiny model and edit tier. Numbers are session IDs.',
                         fig_w - 2 * m_side)
        ax.set_xlim(0, fig_w)  # imshow resets the limits
        ax.set_ylim(0, fig_h)

        if save:
            save_path = save_path or os.path.join(cfg.FIGURES_DIR, 'completed_sessions_vertical.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400)
        if show:
            plt.show()
    return fig


def plot_completed_sessions(master_csv=None, save_path=None, show=True, min_entries=3,
                            save=False, layout=1):
    """Draw which (primary LLM, reasoning effort) x (tiny model, tier) sessions have been run.

    All three layouts are close to 5.2 x 5.0 in with text of 6 to 7.5 pt, so that they stay
    legible when shown small. A label too wide for its tile is shrunk to fit.

    layout=1 is the grid: a row per (primary LLM, effort), the effort under the name, and a
    pair of tier columns per tiny model; the sessions of a cell are stacked, one per line, and
    a row is as tall as its fullest cell; the key is at the bottom. layout=2 has the same
    content drawn in the manner of the bar figures of compare_scores: a panel per tiny model,
    three panels across, and the key in the slot after the last panel. layout=3 is
    layout=2 turned around: a column per (primary LLM, effort), a block of rows per tiny model
    with one row per tier, the sessions of a cell stacked one per line, and the key along the
    bottom.

    A cell is coloured by the best status among its sessions (green: an edit was saved; yellow:
    the reply ran without error but edited nothing; orange: the program raised and nothing was
    edited; red: no valid reply or failed; grey: not run) and
    lists the session IDs. An ID whose status is worse than the cell's is marked with '*'.

    After each ID, a check mark says that the session's edited model has at least `min_entries`
    rows in every results file of its task (GSM8K_500 and MATH500, or BasicArithmetic for
    TinyStories-33M). When it has some rows but not enough, the counts are shown instead, in
    that order, for example '(3/1)'. Nothing is shown when it has none.

    Under each tiny model's name, 'unedited' carries the same mark for the unedited model (the
    rows of 'wsp-<tiny model>-base'), with 'not benchmarked' when it has none. Its task is the
    registry's default task, or that of its first session for a tiny model outside the registry.

    save, save_path: with save=True the figure is written as a PNG to save_path; None means
    completed_sessions.png in the figures folder of the results (completed_sessions_vertical.png
    for layout=2, completed_sessions_transposed.png for layout=3).
    Returns the matplotlib figure.
    """
    _results_models_cache.clear()
    rows, tiny_models, cells = completed_sessions(master_csv)
    if layout == 2:
        return _plot_completed_sessions_vertical(rows, tiny_models, cells, min_entries, save_path, save, show)
    if layout == 3:
        return _plot_completed_sessions_transposed(rows, tiny_models, cells, min_entries, save_path, save, show)
    tiers = cfg.TIERS
    n_tiers = len(tiers)
    info, statuses_shown = _session_cells(cells, min_entries)

    # ~~~ sizes, in inches ~~~
    m_side, m_top, m_bottom = 0.08, 0.40, 0.06
    label_w = 1.22                        # primary LLM name, with its effort under it
    cell_w, pair_gap, tile_gap, line_h = 0.355, 0.06, 0.02, 0.115  # pair_gap: between two tiny models' column pairs
    head_h, key_gap = 0.64, 0.10
    row_h = [max(0.25, 0.04 + line_h * max(
                 [len(info[(model, effort, tiny, tier)][1]) for tiny in tiny_models for tier in tiers
                  if (model, effort, tiny, tier) in info] or [1]))
             for model, effort in rows]
    pair_w = n_tiers * cell_w
    grid_x, grid_top = m_side + label_w, m_top + head_h
    fig_w = grid_x + len(tiny_models) * pair_w + (len(tiny_models) - 1) * pair_gap + m_side
    key_h, key_items = _sessions_key(fig_w - 2 * m_side, _key_colours(statuses_shown, NOT_RUN_COLOUR),
                                     _key_marks(min_entries), stacked=False)
    fig_h = grid_top + sum(row_h) + key_gap + key_h + m_bottom

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': _sessions_sans(),
                         'mathtext.fontset': 'dejavusans'}):
        fig, ax, _text, _tile = _sessions_canvas(fig_w, fig_h, glass=False)

        for j, tiny in enumerate(tiny_models):
            x0 = grid_x + j * (pair_w + pair_gap)
            xc = x0 + pair_w / 2
            # column head, from the bottom up: the tiers, the note on the unedited model, the name
            note, benchmarked = _unedited_note(tiny, cells, min_entries)
            for k, tier in enumerate(tiers):
                _text(x0 + (k + 0.5) * cell_w, grid_top - 0.075, tier.replace('oneshot_', ''),
                      size=6.3, colour=_INK_SOFT, ha='center')
            for n, line in enumerate(reversed(note)):
                _text(xc, grid_top - 0.20 - 0.10 * n, line, size=6.3,
                      colour=_INK_SOFT if benchmarked else _WARN, ha='center', fit=pair_w)
            name = _wrap_at_hyphens(tiny, 13)
            for n, line in enumerate(reversed(name)):
                _text(xc, grid_top - 0.215 - 0.10 * len(note) - 0.115 * n, line, size=7,
                      fontweight='bold', ha='center', fit=pair_w)
            top = grid_top
            for (model, effort), rh in zip(rows, row_h):
                for k, tier in enumerate(tiers):
                    best, lines = info.get((model, effort, tiny, tier), (None, []))
                    colour = STATUS_COLOURS.get(best, OTHER_STATUS_COLOUR) if lines else NOT_RUN_COLOUR
                    x = x0 + k * cell_w
                    _tile(x + tile_gap / 2, top + tile_gap / 2, cell_w - tile_gap, rh - tile_gap, colour)
                    for n, line in enumerate(lines):
                        _text(x + cell_w / 2, top + rh / 2 + (n - (len(lines) - 1) / 2) * line_h, line,
                              size=7.5, ha='center', fit=cell_w - 0.05)
                top += rh

        top = grid_top
        for (model, effort), rh in zip(rows, row_h):
            _text(grid_x - 0.06, top + rh / 2 - 0.05, model, size=7, ha='right', fit=label_w - 0.08)
            _text(grid_x - 0.06, top + rh / 2 + 0.055, effort, size=6.3, colour=_INK_SOFT, ha='right')
            top += rh

        _draw_sessions_key(_text, _tile, m_side, grid_top + sum(row_h) + key_gap, key_items)
        _sessions_titles(_text, m_side, f'{BENCHMARK_NAME} sessions run so far',
                         'Primary LLM and reasoning effort, by tiny model and edit tier. Numbers are session IDs.',
                         fig_w - 2 * m_side)

        if save:
            save_path = save_path or os.path.join(cfg.FIGURES_DIR, 'completed_sessions.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
        if show:
            plt.show()
    return fig


# ============================================================================================
# Which session counts
# ============================================================================================

def configuration_label(model, effort):
    """How a primary LLM at a reasoning effort is named everywhere below: '<model> (<effort>)'."""
    return f'{model} ({effort})'


def _split_label(label):
    """(model, effort) of a configuration label."""
    model, sep, effort = label.rpartition(' (')
    if not sep or not effort.endswith(')'):
        raise ValueError(f"A primary LLM is named '<model> (<reasoning effort>)', for example "
                         f"'gpt-6-astra (max)'; got {label!r}.")
    return model, effort[:-1]


def select_sessions(master_csv=None):
    """The session that counts for each primary LLM, tiny model and tier.

    Returns {configuration label: {tiny model: {tier: session ID}}}, the labels in order of their
    first session. A configuration is a primary LLM at one reasoning effort (the
    `reasoning_effort` of the session's llm_kwargs, or 'default' when none was set).

    The rule: when a configuration has several sessions for the same tiny model and tier, the
    latest one (the largest ID) counts, whatever its status. A session that is run again is
    therefore replaced by its rerun, and an earlier session never counts again.
    """
    rows, _, cells = completed_sessions(master_csv)
    selected = {configuration_label(model, effort): {} for model, effort in rows}
    for (model, effort, tiny_model, tier), sessions in cells.items():
        latest = max(session_id for session_id, _, _ in sessions)
        selected[configuration_label(model, effort)].setdefault(tiny_model, {})[tier] = latest
    return selected


def is_complete(sessions_of_configuration):
    """Whether a configuration has a session for every registered tiny model in every tier."""
    return all(tier in sessions_of_configuration.get(tiny_model, {})
               for tiny_model in cfg.TINY_MODEL_REGISTRY for tier in cfg.TIERS)


def complete_configurations(master_csv=None):
    """The labels of the primary LLMs that have a session for every registered tiny model in
    every tier, in order of their first session. Only these can be given a score."""
    return [label for label, sessions in select_sessions(master_csv).items() if is_complete(sessions)]


def eval_sets_of(tiny_model):
    """The evaluation sets that score a tiny model (those of its default task)."""
    entry = cfg.tiny_entry(tiny_model)
    return tuple(cfg.TASKS[entry['default_task']]['eval_sets'])


# ============================================================================================
# Scores of one tiny model on one evaluation set
# ============================================================================================

# evaluation set -> (the field of its file holding the question, how an answer is compared with
# the reference: 'integer' (equal as integers) or 'math_verify' (mathematically equivalent))
EVAL_SETS = {
    'GSM8K_500': ('question', 'integer'),
    'MATH500': ('problem', 'math_verify'),
    'BasicArithmetic': ('problem', 'integer'),
}
NO_ANSWER = 'NA'  # what a results row holds where no answer could be read from the reply


@dataclass
class Arm:
    """One bar of a comparison: the scoring passes of one model, and how it is named.
    The position of an arm in the list fixes its place in the figure and in the tests."""
    key: str            # short name used in printouts
    df: pl.DataFrame    # its rows of the results file, one per scoring pass
    label: str          # its name in the figure; a second line holds the reasoning effort


_scores_cache = {}  # results file -> (its modification time, its rows)


def _read_scores(eval_set):
    """The rows of an evaluation set's results file. Read again only when the file has changed."""
    path = scores_file(eval_set)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} does not exist: no tiny model has been scored on {eval_set} yet. '
            f'Score the unedited model and the edited checkpoints first (scripts/score_tiny_model.py).')
    modified = os.path.getmtime(path)
    if _scores_cache.get(path, (None, None))[0] != modified:
        _scores_cache[path] = (modified, pl.read_csv(path, infer_schema_length=None))
    return _scores_cache[path][1]


def _reference_answers(eval_set):
    """The reference answers of an evaluation set, in question order: integers, or strings for a
    set graded by math_verify."""
    answers = pl.read_ndjson(os.path.join(cfg.EVAL_DIR, f'{eval_set}_Benchmark.jsonl'))['answer'].to_list()
    if EVAL_SETS[eval_set][1] == 'math_verify':
        return np.array([x for x in answers])
    return np.array([int(x) for x in answers])


def _question_scores(arm, eval_set, references):
    """The 0/1 score of every question in every scoring pass of an arm: an array of shape
    (passes, questions).

    A missing answer is wrong. In the integer sets it is read as 0 before the comparison, so it
    counts as correct for a question whose reference answer is 0.
    """
    answers = arm.df['model_answers']
    if EVAL_SETS[eval_set][1] == 'math_verify':
        answers = np.stack(answers.str.json_decode(pl.List(pl.String)).to_numpy())
        return np.array([[0 if str(answer) == NO_ANSWER
                          else _math_verify_correct(str(references[q]), str(answer))
                          for q, answer in enumerate(run)] for run in answers])
    answers = np.stack(answers.str.replace_all(f'"{NO_ANSWER}"', '0', literal=True)
                       .str.json_decode(pl.List(pl.Int64)).to_numpy())
    return np.array([[int(answer == references[q]) for q, answer in enumerate(run)] for run in answers])


def paired_difference(scores_a, scores_b, confidence=0.95):
    """The difference in mean score between two arms scored on the same questions, A minus B,
    with its paired standard error.

    scores_a, scores_b: 0/1 arrays of shape (passes, questions). Each question's score is first
    averaged over the passes of an arm; the difference of those per-question means is then
    treated as one sample per question, so its standard error is the standard deviation of the
    per-question differences divided by the square root of the number of questions.

    Returns {'diff', 'se', 'ci' (lower, upper), 'z', 'p'}; p is two-sided, from the normal
    distribution.
    """
    per_question = np.mean(scores_a, axis=0) - np.mean(scores_b, axis=0)
    n = len(per_question)
    diff = np.mean(per_question)
    se = np.sqrt((1 / n) * (1 / (n - 1) * np.sum((per_question - diff) ** 2)))
    half_width = abs(stats.norm.ppf((1 - confidence) / 2)) * se
    z = np.abs(diff) / se
    return {'diff': diff, 'se': se, 'ci': (diff - half_width, diff + half_width),
            'z': z, 'p': 2 * (1 - stats.norm.cdf(z))}


def holm_rejections(p_values, alpha):
    """Holm-Bonferroni step-down over a list of p-values: which hypotheses are rejected at
    family-wise level alpha. The smallest p is tested at alpha / m, the next at alpha / (m - 1),
    and so on; testing stops at the first p that fails."""
    order = sorted(range(len(p_values)), key=lambda k: p_values[k])
    rejected, rejecting = [False] * len(p_values), True
    for step, k in enumerate(order):
        rejecting = rejecting and (p_values[k] <= alpha / (len(p_values) - step))
        rejected[k] = rejecting
    return rejected


def cld_letters(sig_pairs, means):
    """Compact letter display by insert-and-absorb (Piepho, 2004): groups that share a letter do
    not differ significantly.
    sig_pairs: (i, j) index pairs judged significantly different.
    means: the mean of each group, used only to order the letters ('a' holds the best group)."""
    cols = [frozenset(range(len(means)))]
    for i, j in sig_pairs:
        for col in [c for c in cols if {i, j} <= c]:            # insert
            cols.remove(col)
            cols += [col - {i}, col - {j}]
        cols = [c for c in set(cols)                            # absorb
                if not any(c < d for d in cols)]
    cols.sort(key=lambda c: sorted(-means[g] for g in c))
    return [''.join(chr(97 + k) for k, c in enumerate(cols) if g in c)
            for g in range(len(means))]


# MATH500 answers are graded by math_verify in a worker process (llm/math_grader.py). Starting it
# takes seconds, so one grader serves the whole Python session: it is started by the first answer
# that needs grading (never by importing this module, and never for the other evaluation sets),
# reused by every later call, and closed when the interpreter exits. Each (reference answer, model
# answer) pair is graded once and remembered.
_math_grades = {}   # (reference answer, model answer) -> 1 / 0
_math_grader = []   # the one MathVerifyGrader, once it has been started


def _close_math_grader():
    """Stop the grader's worker process, if one was started. Registered to run at interpreter exit."""
    while _math_grader:
        try:
            _math_grader.pop().close()
        except Exception:
            pass


def _math_verify_correct(reference, answer):
    """1 if math_verify judges `answer` equivalent to `reference` (both strings), else 0."""
    key = (reference, answer)
    if key not in _math_grades:
        for attempt in range(2):
            if not _math_grader:
                from wspmathbench.llm.math_grader import MathVerifyGrader
                if not _math_grades:  # the first grader of the session
                    atexit.register(_close_math_grader)
                _math_grader.append(MathVerifyGrader(timeout=10))
            try:
                _math_grades[key] = int(_math_grader[0].grade(reference, answer) == 'correct')
                break
            except (OSError, ValueError) as e:
                # a ValueError naming the gold answer is math_verify failing to parse the
                # reference, which a new worker would not change
                if attempt or 'gold answer' in str(e):
                    raise
                _math_grader.clear()  # the worker died between calls: start a new one and retry once
    return _math_grades[key]


def _arms_of_primary_llm(df, model, tiny_model, tier, master_csv):
    """The unedited tiny model, then the model edited by `model` in one tier or in both."""
    sessions = select_sessions(master_csv).get(model, {}).get(tiny_model, {})
    tiers = list(cfg.TIERS) if tier == 'both' else [tier]
    arms = [Arm('unedited', df.filter(pl.col('model') == scored_model_key(tiny_model)), 'Unedited')]
    for t in tiers:
        if t not in sessions:
            raise ValueError(f'{model} has no {t} session for {tiny_model} in the master table.')
        arms.append(Arm(t, df.filter(pl.col('model') == scored_model_key(tiny_model, sessions[t])), t))
    return arms


def _arms_of_tiny_model(df, tiny_model, tier, primary_llms, master_csv):
    """The unedited tiny model, then one arm per primary LLM that edited it in `tier`, in order
    of their first session for this tiny model."""
    selected = select_sessions(master_csv)
    if primary_llms is None:
        primary_llms = [label for label, sessions in selected.items() if is_complete(sessions)]
        if not primary_llms:  # nothing is complete yet: show whatever has a session here
            primary_llms = [label for label, sessions in selected.items() if tier in sessions.get(tiny_model, {})]
    first_session = {}  # configuration label -> its earliest session ID for this tiny model
    for (m, e, tiny, _), sessions in completed_sessions(master_csv)[2].items():
        if tiny == tiny_model:
            label = configuration_label(m, e)
            first_session[label] = min([first_session.get(label, math.inf)] + [s[0] for s in sessions])
    arms = [Arm('unedited', df.filter(pl.col('model') == scored_model_key(tiny_model)), 'Unedited')]
    for label in sorted(primary_llms, key=lambda l: first_session.get(l, math.inf)):
        sessions = selected.get(label, {}).get(tiny_model, {})
        if tier not in sessions:
            raise ValueError(f'{label} has no {tier} session for {tiny_model} in the master table.')
        model, effort = _split_label(label)
        arms.append(Arm(label, df.filter(pl.col('model') == scored_model_key(tiny_model, sessions[tier])),
                        f'{model}\n{effort}'))
    return arms


def compare_scores(model=None, tiny_model='Qwen3.5-0.8B', benchmark=None, tier='oneshot_spec',
                   primary_llms=None, confidence=0.95, show_cld=True, ylim=None,
                   plot=True, show=True, save=False, verbose=True, master_csv=None):
    """Compare an unedited tiny model with its edited versions on one evaluation set.

    Two kinds of comparison:
      model given    the unedited tiny model against the model that primary LLM edited, in one
                     tier or, with tier='both', in each tier. This is the comparison the
                     WSPMathBench score is built from.
      model=None     the unedited tiny model against the model edited by each primary LLM in
                     `tier`: by default every primary LLM with a complete set of sessions (see
                     complete_configurations), or those named in `primary_llms`.

    model, primary_llms : configuration labels, '<model> (<reasoning effort>)'.
    benchmark : an evaluation set of the tiny model ('GSM8K_500' or 'MATH500', or
        'BasicArithmetic' for TinyStories-33M); None means its first.
    confidence : the family-wise confidence level of the tests.
    plot, show, save : draw the bar figure; show it; save it as PDF and PNG under the figures
        folder of the results. ylim, a [low, high] list, fixes the score axis.

    The session that counts for a primary LLM, tiny model and tier is the latest one
    (select_sessions). Each arm is every scoring pass of its model in the results file.

    Statistics. An arm's score is the mean over its scoring passes of the fraction of questions
    answered correctly. Every pair of arms is compared question by question (paired_difference),
    the p-values are corrected by Holm-Bonferroni, and the arms get letters such that arms
    sharing a letter do not differ significantly (cld_letters).

    Returns a dict:
      arm_keys, arm_IDs      the arms in order, and the results-row IDs of each
      score_means, score_stds, score_sems   per arm, over its scoring passes
      cld                    the letter(s) of each arm
      mean_diff_list, SE_mean_diff_list, CI_mean_diff_list
                             [i][k]: arm i minus the k-th of the other arms (in arm order, arm i
                             left out), its standard error and its confidence interval. With
                             two arms, [1][0] is edited minus unedited.
      figure                 the matplotlib figure, or None
    """
    eval_sets = eval_sets_of(tiny_model)
    benchmark = eval_sets[0] if benchmark is None else benchmark
    if benchmark not in eval_sets:
        raise ValueError(f'{tiny_model} is scored on {list(eval_sets)}, not on {benchmark!r}.')
    if model is None and tier not in cfg.TIERS:
        raise ValueError(f'`tier` must be one of {cfg.TIERS} when `model` is None, got {tier!r}.')
    if model is not None and tier not in cfg.TIERS + ('both',):
        raise ValueError(f"`tier` must be one of {cfg.TIERS + ('both',)}, got {tier!r}.")

    df = _read_scores(benchmark)
    if model is not None:
        arms = _arms_of_primary_llm(df, model, tiny_model, tier, master_csv)
    else:
        arms = _arms_of_tiny_model(df, tiny_model, tier, primary_llms, master_csv)
    empty = [arm.key for arm in arms if arm.df.height == 0]
    if empty:
        raise ValueError(f'{scores_file(benchmark)} has no scoring pass of {tiny_model} for: '
                         f'{", ".join(empty)}. Score those models first.')

    references = _reference_answers(benchmark)
    groups = [arm.df['score'].to_numpy() for arm in arms]
    means = np.array([g.mean() for g in groups])
    sds = np.array([g.std(ddof=1) for g in groups])
    sems = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in groups])

    # ~~~ pairwise per-question tests, Holm step-down, letters ~~~
    question_scores = [_question_scores(arm, benchmark, references) for arm in arms]
    pair = {(i, j): paired_difference(question_scores[i], question_scores[j], confidence)
            for i in range(len(arms)) for j in range(len(arms)) if i != j}
    others = [[j for j in range(len(arms)) if j != i] for i in range(len(arms))]
    tested = [(i, j) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    rejected = holm_rejections([pair[ij]['p'] for ij in tested], 1 - confidence)
    letters = cld_letters([ij for ij, r in zip(tested, rejected) if r],
                          [np.mean(s) for s in question_scores])
    if verbose:
        for arm, mean, letter in zip(arms, means, letters):
            print(f'  {arm.key:32s} n={arm.df.height}  mean={mean:.4f}  ->  {letter}')

    figure = None
    if plot:
        figure = _plot_score_bars(arms, groups, means, sems, letters if show_cld else [], benchmark,
                                  tiny_model, model, tier, confidence, ylim, save, show)
    return {
        'model_label': model,
        'tiny_model': tiny_model, 'benchmark': benchmark, 'tier': tier, 'confidence': confidence,
        'arm_keys': [arm.key for arm in arms],
        'arm_IDs': [arm.df['ID'].to_list() for arm in arms],
        'score_means': means, 'score_stds': sds, 'score_sems': sems,
        'cld': letters,
        'mean_diff_list': [[pair[(i, j)]['diff'] for j in others[i]] for i in range(len(arms))],
        'SE_mean_diff_list': [[pair[(i, j)]['se'] for j in others[i]] for i in range(len(arms))],
        'CI_mean_diff_list': [[pair[(i, j)]['ci'] for j in others[i]] for i in range(len(arms))],
        'figure': figure,
    }


# ============================================================================================
# The bar figure of one comparison
# ============================================================================================

_BAR_INK, _BAR_INK_SOFT, _BAR_INK_FAINT = '#1D1D24', '#55555F', '#8A8A96'
_UNEDITED_COLOUR, _EDITED_COLOUR = '#7A7A86', '#4B4BF2'


def _sans_fonts():
    """The sans-serif fonts of the figures, best first, of those installed."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ('Inter', 'Segoe UI', 'Helvetica Neue', 'Arial') if f in installed] + ['DejaVu Sans']


def _pin_fonts(fig, fonts):
    """Name the font family on every text of a figure, so that a figure drawn inside an rc
    context keeps its fonts when it is drawn or saved again outside it."""
    for text in fig.findobj(Text):
        text.set_fontfamily(fonts)


def _wrap_name(name, width=16):
    """A long model name broken at its hyphens into lines of at most `width` characters."""
    lines, cur = [], ''
    for part in name.split('-'):
        trial = part if cur == '' else cur + '-' + part
        if len(trial) > width and cur != '':
            lines.append(cur + '-')
            cur = part
        else:
            cur = trial
    return lines + [cur]


def _draw_glass_bar(ax, xc, h, bar_w, base, colour, rx, ry):
    """A bar from `base` to `h` (up or down) centred on xc, with rounded corners at the end
    away from the base: a vertical fade of the colour, a soft white band down the left side, a
    coloured rim and a white inner highlight. rx, ry: the corner radius in data units."""
    x0, x1 = xc - bar_w / 2, xc + bar_w / 2
    path = Path([(x0, base), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h),
                 (x1, h), (x1, h - ry), (x1, base), (x0, base)],
                [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
                 Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY])
    clip = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
    ax.add_patch(clip)
    extent = (x0, x1, base, h)
    # vertical fade: saturated at the far end, pale and translucent at the base
    fade = np.linspace(0, 1, 256)[:, None]
    body = np.empty((256, 1, 4))
    body[..., :3] = (1 - fade[..., None]) * (colour + (1 - colour) * 0.62) + fade[..., None] * colour
    body[..., 3] = 0.50 + 0.48 * fade ** 0.8
    ax.imshow(body, extent=extent, origin='lower', aspect='auto',
              interpolation='bicubic', zorder=2).set_clip_path(clip)
    across = np.linspace(0, 1, 256)[None, :]
    sheen = np.ones((1, 256, 4))
    sheen[..., 3] = 0.30 * np.clip(1 - across / 0.55, 0, 1) ** 1.6
    ax.imshow(sheen, extent=extent, origin='lower', aspect='auto',
              interpolation='bicubic', zorder=2.1).set_clip_path(clip)
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor=(*colour, 0.60),
                           linewidth=0.7, zorder=2.3, clip_on=False))
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor=(1, 1, 1, 0.55),
                           linewidth=1.6, zorder=2.2)).set_clip_path(clip)


def _save_figure(fig, name):
    """Save a figure as <figures folder>/<name>.pdf and .png. Returns the PDF's path."""
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    name = name.replace(' ', '_')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(cfg.FIGURES_DIR, f'{name}.{ext}'))
    print(f'Saved {name}.pdf and {name}.png to {cfg.FIGURES_DIR}')
    return os.path.join(cfg.FIGURES_DIR, name + '.pdf')


def _plot_score_bars(arms, groups, means, sems, letters, benchmark, tiny_model, model, tier,
                     confidence, ylim, save, show):
    """The bar figure of compare_scores. Rounded bars of the mean score, grey for the unedited
    arm and one accent colour for the others. Above each bar: its letter. On each bar: its
    scoring passes (dots on the bar's shoulders) and the standard error of the mean (centre
    whisker). Under the axis: the mean score, the arm's name, then its reasoning effort and
    number of passes. Sized for a paper: about 0.86 in per arm, text of 6.5 pt and up."""
    ink, ink_soft, ink_faint = _BAR_INK, _BAR_INK_SOFT, _BAR_INK_FAINT
    n_arms = len(arms)
    letter_of = dict(enumerate(letters))
    fonts = _sans_fonts()

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': fonts,
                         'mathtext.fontset': 'dejavusans'}):
        # ~~~ layout: fixed margins (inches), so the data -> pixel scale is known before
        # anything is drawn ~~~
        fig_w, fig_h = max(3.4, 0.86 + 0.86 * n_arms), 3.7
        m_left, m_right, m_top, m_bottom = 0.66, 0.12, 0.62, 1.10
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor='white')
        ax = fig.add_axes([m_left / fig_w, m_bottom / fig_h,
                           1 - (m_left + m_right) / fig_w, 1 - (m_top + m_bottom) / fig_h])
        pt = fig.dpi / 72  # points -> pixels
        ax_h_px = (fig_h - m_top - m_bottom) * fig.dpi

        marks_top = np.array([max(means[k] + sems[k], np.max(groups[k])) for k in range(n_arms)])
        if type(ylim) == list and len(ylim) == 2:
            y_lo, y_hi = ylim
        else:
            # bars start at zero; the top leaves room for the letter above the highest mark
            y_lo = 0
            y_hi = max(np.max(marks_top) * ax_h_px / (ax_h_px - (22 if letter_of else 8) * pt), 0.02)
        bar_w = 0.68
        # a slot per arm, padded at the sides when there are few arms so the bars stay slim
        slots = max(n_arms + 0.16, (fig_w - m_left - m_right) / 0.90)
        x_lims = ((n_arms - 1) / 2 - slots / 2, (n_arms - 1) / 2 + slots / 2)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lo, y_hi)
        px0 = ax.transData.transform((0, y_lo))
        px1 = ax.transData.transform((1, y_lo + 1))
        px_x, px_y = px1[0] - px0[0], px1[1] - px0[1]  # pixels per data unit
        r_px = 0.20 * bar_w * px_x  # corner radius, in pixels so the corners stay circular

        def _shoulder_offsets(g):
            '''x offsets that seat the passes on the two shoulders of the bar, alternating sides
            in order of score, so the centre stays clear for the whisker and the letter; passes
            at (nearly) the same height on one side take neighbouring columns'''
            columns = (0.25, 0.19, 0.31)
            offsets, placed = np.zeros(len(g)), {-1: [], 1: []}
            for rank, j in enumerate(np.argsort(g, kind='stable')):
                side = -1 if rank % 2 == 0 else 1
                y_px = g[j] * px_y
                free = [col for col in columns
                        if all(c != col or abs(y_px - y) > 4.5 * pt for y, c in placed[side])]
                col = free[0] if free else columns[len(placed[side]) % len(columns)]
                placed[side].append((y_px, col))
                offsets[j] = side * col
            return offsets

        for i in range(n_arms):
            c = np.array(to_rgb(_UNEDITED_COLOUR if arms[i].key == 'unedited' else _EDITED_COLOUR))
            h = means[i]
            if h > y_lo:
                _draw_glass_bar(ax, i, h, bar_w, y_lo, c, r_px / px_x, min(r_px / px_y, h - y_lo))
            else:  # a zero score: leave a hairline so the arm is visibly present
                ax.plot([i - bar_w / 2, i + bar_w / 2], [y_lo, y_lo], color=c, linewidth=2,
                        solid_capstyle='round', zorder=2, clip_on=False)

            # the scoring passes behind the mean, and its standard error
            g = groups[i]
            ax.scatter(i + _shoulder_offsets(g), g, s=11, zorder=4, facecolor='white',
                       edgecolor=c * 0.70, linewidth=0.8, clip_on=False)
            if sems[i] > 0:
                ax.plot([i, i], [h - sems[i], h + sems[i]], color=ink, linewidth=1.1,
                        solid_capstyle='round', zorder=3,
                        path_effects=[pe.withStroke(linewidth=2.6, foreground=(1, 1, 1, 0.85))])

            # the letter, above everything drawn for the arm
            if i in letter_of:
                ax.annotate(letter_of[i], xy=(i, marks_top[i]), xytext=(0, 4.5),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=10.5, fontweight='bold', color=ink, zorder=5,
                            annotation_clip=False)

            # under the axis: the mean, the arm's name, then the effort and number of passes
            name, _, sub = arms[i].label.partition('\n')
            sub = (sub + ' · ' if sub else '') + f'n={len(g)}'
            rows = [(f'{100 * h:.1f}', ink, 7.5, 'semibold')]
            rows += [(line, ink, 7, 'normal') for line in _wrap_name(name)]
            rows += [(sub, ink_faint, 6.5, 'normal')]
            y_pt = -4.5
            for text, colour, size, weight in rows:  # one text per line keeps the rows aligned across arms
                y_pt -= size + (3.5 if weight == 'semibold' or colour == ink_faint else 2)
                ax.annotate(text, xy=(i, 0), xycoords=('data', 'axes fraction'),
                            xytext=(0, y_pt), textcoords='offset points', ha='center',
                            va='baseline', fontsize=size, fontweight=weight, color=colour,
                            annotation_clip=False)
                if weight == 'semibold':
                    y_pt -= 2.5
        # name the row of means, in line with the y tick labels
        ax.annotate('mean', xy=(0, 0), xycoords='axes fraction', xytext=(-4, -15.5),
                    textcoords='offset points', ha='right', va='baseline',
                    fontsize=6.5, color=ink_faint, annotation_clip=False)

        ax.set_xlim(x_lims)  # imshow resets the limits
        ax.set_ylim(y_lo, y_hi)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{100 * v:g}'))
        ax.set_ylabel(f"{benchmark.replace('_', '-')} score (%)", fontsize=8, color=ink_soft, labelpad=5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E6E6EB', linewidth=0.6)
        ax.tick_params(axis='y', length=0, labelsize=7.5, labelcolor=ink_soft, pad=4)
        ax.tick_params(axis='x', length=0, labelbottom=False)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B9B9C4')
        ax.spines['bottom'].set_linewidth(0.8)

        # ~~~ title, subtitle, footnote ~~~
        if model is None:
            title = f'{tiny_model} after {tier} edits'
            subtitle = f'{BENCHMARK_NAME}: one bar per editing LLM'
        else:
            model_name, effort = _split_label(model)
            title = f'{tiny_model}, edits by {model_name}'
            subtitle = f'{BENCHMARK_NAME}: by edit tier, {effort} effort'
        x_text = 0.10 / fig_w
        fig.text(x_text, 1 - 0.20 / fig_h, title, fontsize=10, fontweight='bold',
                 color=ink, ha='left', va='center')
        fig.text(x_text, 1 - 0.40 / fig_h, subtitle,
                 fontsize=7.5, color=ink_soft, ha='left', va='center')
        note = 'Bar: mean score. Whisker: ±SEM. Dots: individual runs.'
        if model is None:
            note += ' Under each LLM: reasoning effort and number of runs.'
        if letter_of:
            note += (f'\nArms sharing a letter do not differ significantly '
                     f'(paired per-question tests, Holm-Bonferroni correction, {int(confidence * 100)}% CL).')
        per_line = int((fig_w - 0.2) / (6.5 * 0.5 / 72))  # characters that fit across the figure
        note = '\n'.join(textwrap.fill(part, per_line) for part in note.split('\n'))
        fig.text(x_text, 0.08 / fig_h, note, fontsize=6.5, color=ink_faint,
                 ha='left', va='bottom', linespacing=1.3)
        _pin_fonts(fig, fonts)

        if save:
            who = tier if model is None else model
            name = f'bars_{tiny_model}_{benchmark}_{who}'
            if confidence != 0.95:
                name += f'_{confidence}CL'
            _save_figure(fig, name)
        if show:
            plt.show()
    return fig


# ============================================================================================
# The WSPMathBench score
# ============================================================================================

def missing_scores(model, tier, master_csv=None):
    """What still has to be scored before `model` can be given a score in `tier`: a list of
    (tiny model, evaluation set, results-file model key) with no scoring pass. Empty when the
    score can be computed. A tiny model with no session at all is listed with key None."""
    sessions = select_sessions(master_csv).get(model, {})
    missing = []
    for tiny_model in cfg.TINY_MODEL_REGISTRY:
        for eval_set in eval_sets_of(tiny_model):
            if tier not in sessions.get(tiny_model, {}):
                missing.append((tiny_model, eval_set, None))
                continue
            scored = (_read_scores(eval_set)['model'].to_list() if os.path.exists(scores_file(eval_set)) else [])
            for key in (scored_model_key(tiny_model), scored_model_key(tiny_model, sessions[tiny_model][tier])):
                if key not in scored:
                    missing.append((tiny_model, eval_set, key))
    return missing


def wsp_score(model, tier='oneshot_spec', confidence=0.95, verbose=True, master_csv=None):
    """The WSPMathBench score of one primary LLM in one tier, with its standard error.

    model: a configuration label, '<model> (<reasoning effort>)'.

    For every tiny model and each of its evaluation sets, the edited model is compared with the
    unedited one (compare_scores). The change in score, edited minus unedited, is counted as it
    is when it is significant at `confidence` and as zero when it is not; an edit that made the
    model significantly worse counts with its negative sign. The changes are averaged over the
    evaluation sets of a tiny model, so that a tiny model with one evaluation set weighs as much
    as one with two, and then over the tiny models.

    Uncertainty. Each change has the paired per-question standard error of compare_scores. The
    score is a mean of means of those changes, so with the evaluation sets taken as independent
    (different questions, different tiny models) its standard error follows by adding variances:
      SE(tiny model) = sqrt(sum of SE^2 over its evaluation sets) / number of its evaluation sets
      SE(score)      = sqrt(sum of SE(tiny model)^2) / number of tiny models
    A change counted as zero for not being significant keeps its standard error: the underlying
    change is just as uncertain as one that is counted.

    Returns {'score', 'score_se', 'tiny_models', 'signif_mean_diffs', 'signif_mean_diffs_se'};
    the last two hold one entry per tiny model, in the order of 'tiny_models'. Scores are
    fractions: multiply by 100 for points.
    """
    tiny_models = list(cfg.TINY_MODEL_REGISTRY)
    counted, counted_se = [], []
    for tiny_model in tiny_models:
        changes, changes_se = [], []
        for eval_set in eval_sets_of(tiny_model):
            result = compare_scores(model=model, tiny_model=tiny_model, tier=tier, benchmark=eval_set,
                                    confidence=confidence, plot=False, verbose=False, master_csv=master_csv)
            unedited_letter, edited_letter = result['cld']
            significant = unedited_letter != edited_letter
            changes.append(result['mean_diff_list'][1][0] if significant else 0)
            changes_se.append(result['SE_mean_diff_list'][1][0])
        counted.append(np.mean(changes))
        counted_se.append(np.sqrt(np.sum(np.square(changes_se))) / len(changes_se))
    score = np.mean(counted)
    score_se = np.sqrt(np.sum(np.square(counted_se))) / len(counted_se)
    if verbose:
        print(f'{model}, {tier}: {100 * score:.2f} +/- {100 * score_se:.2f} points')
    return {'score': score, 'score_se': score_se, 'tiny_models': tiny_models,
            'signif_mean_diffs': counted, 'signif_mean_diffs_se': counted_se}


def compile_scores(primary_llms=None, tiers=None, confidence=0.95, save=True, verbose=True, master_csv=None):
    """The WSPMathBench score of every primary LLM in every tier, as one table.

    primary_llms: configuration labels; None means every primary LLM with a session for every
        tiny model in every tier (complete_configurations), in order of first session.
    tiers: None means cfg.TIERS.

    A primary LLM and tier whose tiny models have not all been scored yet is left out, with a
    printed list of what is missing. Returns a polars DataFrame with the columns model, tier,
    score, score_se, tiny_models, signif_mean_diffs and signif_mean_diffs_se (see wsp_score),
    and with save=True writes it to cfg.COMPILED_METRICS_FILE.
    """
    primary_llms = complete_configurations(master_csv) if primary_llms is None else list(primary_llms)
    tiers = list(cfg.TIERS) if tiers is None else list(tiers)
    records = []
    for model in primary_llms:
        for tier in tiers:
            missing = missing_scores(model, tier, master_csv)
            if missing:
                if verbose:
                    print(f'{model}, {tier}: left out, not scored yet: '
                          + '; '.join(f"{tiny_model} on {eval_set} ({key or 'no session'})"
                                      for tiny_model, eval_set, key in missing))
                continue
            result = wsp_score(model, tier, confidence=confidence, verbose=verbose, master_csv=master_csv)
            records.append({'model': model, 'tier': tier, 'score': float(result['score']),
                            'score_se': float(result['score_se']), 'tiny_models': result['tiny_models'],
                            'signif_mean_diffs': [float(x) for x in result['signif_mean_diffs']],
                            'signif_mean_diffs_se': [float(x) for x in result['signif_mean_diffs_se']]})
    schema = {'model': pl.String, 'tier': pl.String, 'score': pl.Float64, 'score_se': pl.Float64,
              'tiny_models': pl.List(pl.String), 'signif_mean_diffs': pl.List(pl.Float64),
              'signif_mean_diffs_se': pl.List(pl.Float64)}
    df = pl.DataFrame(records, schema=schema)
    if save and df.height:
        os.makedirs(os.path.dirname(cfg.COMPILED_METRICS_FILE), exist_ok=True)
        df.write_parquet(cfg.COMPILED_METRICS_FILE)
        if verbose:
            print(f'Wrote {cfg.COMPILED_METRICS_FILE}')
    return df


def plot_scores(file=None, tier='both', save=False, show=True):
    """Bar figure of the WSPMathBench scores in the file written by compile_scores (default:
    cfg.COMPILED_METRICS_FILE). One group of bars per primary LLM, in the order of the file,
    with one bar per tier. The score is written past each bar, in points (100 x the score in
    the file), with its standard error, and a whisker spans the score plus and minus that
    standard error. tier='both' shows every tier in the file; tier='oneshot_spec' or
    tier='oneshot_code' shows that tier alone, in the colour it has in the figure of both.
    With save=True the figure goes to the figures folder as PDF and PNG. Returns the figure."""
    file = cfg.COMPILED_METRICS_FILE if file is None else file
    df = pl.read_parquet(file)
    models = list(dict.fromkeys(df['model'].to_list()))  # in the order of the file
    tiers = list(dict.fromkeys(df['tier'].to_list()))
    score_of = {(m, t): s for m, t, s in zip(df['model'], df['tier'], df['score'])}
    se_of = {(m, t): s for m, t, s in zip(df['model'], df['tier'], df['score_se'])}
    n_tiny = df['signif_mean_diffs'].list.len().max()

    ink, ink_soft, ink_faint = _BAR_INK, _BAR_INK_SOFT, _BAR_INK_FAINT
    tier_colours = ['#4B4BF2', '#F2784B', '#12A594', '#B14BF2']
    colour_of = {t: np.array(to_rgb(tier_colours[k % len(tier_colours)])) for k, t in enumerate(tiers)}
    if tier not in (None, 'both'):  # one tier alone; it keeps the colour it has beside the others
        if tier not in tiers:
            raise ValueError(f'tier={tier!r} is not in {file}; it holds {tiers}')
        tiers = [tier]
        score_of = {key: s for key, s in score_of.items() if key[1] == tier}
    fonts = _sans_fonts()

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': fonts,
                         'mathtext.fontset': 'dejavusans'}):
        # ~~~ layout: fixed margins (inches), so the data -> pixel scale is known before drawing ~~~
        n_models, n_tiers = len(models), len(tiers)
        slot_in = max(0.92, 0.50 + 0.26 * n_tiers)  # width given to a primary LLM, wide enough for its name
        fig_w, fig_h = max(3.4, 0.86 + slot_in * n_models), 3.7
        m_left, m_right, m_top, m_bottom = 0.66, 0.12, 0.74, 0.98
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor='white')
        ax = fig.add_axes([m_left / fig_w, m_bottom / fig_h,
                           1 - (m_left + m_right) / fig_w, 1 - (m_top + m_bottom) / fig_h])
        pt = fig.dpi / 72  # points -> pixels
        ax_h_px = (fig_h - m_top - m_bottom) * fig.dpi

        # the bars start at zero; each end of the axis leaves room for the score and its standard
        # error, written past the bar and its whisker
        ends = [s + sign * se_of.get(key, 0) for key, s in score_of.items() for sign in (-1, 1)]
        label_px = 20 * pt
        top, low = max(max(ends), 0), min(min(ends), 0)
        if top == low:
            top = 0.01
        span = (top - low) * ax_h_px / (ax_h_px - label_px * (1 + (low < 0)))  # data units over the axis height
        y_hi = top + label_px * span / ax_h_px
        y_lo = low - (label_px * span / ax_h_px if low < 0 else 0)
        slots = max(n_models + 0.10, (fig_w - m_left - m_right) / (slot_in + 0.06))
        x_lims = ((n_models - 1) / 2 - slots / 2, (n_models - 1) / 2 + slots / 2)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lo, y_hi)
        px0, px1 = ax.transData.transform((0, 0)), ax.transData.transform((1, 1))
        px_x, px_y = px1[0] - px0[0], px1[1] - px0[1]  # pixels per data unit
        group_w = 0.80 if n_tiers > 1 else 0.68  # width of a model's group of bars, in slots
        bar_w = group_w / n_tiers * 0.90
        r_px = 0.20 * bar_w * px_x  # corner radius, in pixels so the corners stay circular

        for i, m in enumerate(models):
            for k, t in enumerate(tiers):
                if (m, t) not in score_of:
                    continue
                h, c = score_of[(m, t)], colour_of[t]
                xc = i + (k - (n_tiers - 1) / 2) * group_w / n_tiers
                if abs(h) * px_y > 1.5 * pt:
                    _draw_glass_bar(ax, xc, h, bar_w, 0, c, r_px / px_x,
                                    np.sign(h) * min(r_px / px_y, abs(h)))
                else:  # a score of (nearly) zero: leave a hairline so the bar is visibly present
                    ax.plot([xc - bar_w / 2, xc + bar_w / 2], [0, 0], color=c, linewidth=2,
                            solid_capstyle='round', zorder=2, clip_on=False)
                # the standard error of the score: a whisker of one standard error either side
                se = se_of.get((m, t), 0)
                if se > 0:
                    ax.plot([xc, xc], [h - se, h + se], color=ink, linewidth=1.1, solid_capstyle='round',
                            zorder=3, clip_on=False,
                            path_effects=[pe.withStroke(linewidth=2.6, foreground=(1, 1, 1, 0.85))])
                # the score, past the end of the bar and its whisker, with the standard error nearest the whisker
                up = h >= 0
                lines = [(f'±{100 * se:.1f}', 5.5, 'normal', ink_soft),
                         (f'{round(100 * h, 1) + 0.0:.1f}', 6.5, 'semibold', ink)]
                y_pt = 3.5
                for text, size, weight, colour in lines:
                    ax.annotate(text, xy=(xc, h + se if up else h - se), xytext=(0, y_pt if up else -y_pt),
                                textcoords='offset points', ha='center', va='bottom' if up else 'top',
                                fontsize=size, fontweight=weight, color=colour, zorder=5, annotation_clip=False)
                    y_pt += size + 1.5

            # under the axis: the primary LLM, then its reasoning effort
            name, effort = _split_label(m)
            rows = [(line, ink, 7) for line in _wrap_name(name)] + [(effort, ink_faint, 6.5)]
            y_pt = -3
            for text, colour, size in rows:  # one text per line keeps the rows aligned across models
                y_pt -= size + (3.5 if colour == ink_faint else 2)
                ax.annotate(text, xy=(i, 0), xycoords=('data', 'axes fraction'), xytext=(0, y_pt),
                            textcoords='offset points', ha='center', va='baseline', fontsize=size,
                            color=colour, annotation_clip=False)

        ax.set_xlim(x_lims)  # imshow resets the limits
        ax.set_ylim(y_lo, y_hi)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{100 * v:g}'))
        ax.set_ylabel(f'{BENCHMARK_NAME} score (points)', fontsize=8, color=ink_soft, labelpad=5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E6E6EB', linewidth=0.6)
        ax.tick_params(axis='y', length=0, labelsize=7.5, labelcolor=ink_soft, pad=4)
        ax.tick_params(axis='x', length=0, labelbottom=False)
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.axhline(0, color='#B9B9C4', linewidth=0.8, zorder=1.5)

        # ~~~ title, subtitle, key of the tiers, footnote ~~~
        x_text = 0.10 / fig_w
        fig.text(x_text, 1 - 0.20 / fig_h, f'{BENCHMARK_NAME} score by primary LLM', fontsize=10,
                 fontweight='bold', color=ink, ha='left', va='center')
        fig.text(x_text, 1 - 0.40 / fig_h,
                 'One group of bars per primary LLM, one bar per tier' if n_tiers > 1
                 else f'One bar per primary LLM, {tiers[0]} edits',
                 fontsize=7.5, color=ink_soft, ha='left', va='center')
        full = fig.add_axes([0, 0, 1, 1], facecolor='none')  # for the key, in inches
        full.set_xlim(0, fig_w)
        full.set_ylim(0, fig_h)
        full.axis('off')
        x_key = m_left
        for t in tiers:
            _glass_tile(full, x_key, fig_h - 0.635, 0.11, 0.11, colour_of[t])
            full.text(x_key + 0.16, fig_h - 0.58, t, fontsize=7, color=ink_soft, ha='left', va='center')
            x_key += 0.16 + 7 * 0.60 / 72 * len(t) + 0.18
        note = (f'Score: the change in benchmark score from the unedited to the edited tiny model, '
                f'in percentage points, counted as zero where the change is not significant, '
                f'averaged over the benchmarks of a tiny model and then over the {n_tiny} tiny models.'
                f' Whisker and ±: one standard error of the score, from the paired per-question '
                f'standard errors of every benchmark, taken as independent.')
        per_line = int((fig_w - 0.2) / (6.5 * 0.53 / 72))  # characters that fit across the figure
        fig.text(x_text, 0.08 / fig_h, textwrap.fill(note, per_line), fontsize=6.5, color=ink_faint,
                 ha='left', va='bottom', linespacing=1.3)
        _pin_fonts(fig, fonts)

        if save:
            _save_figure(fig, 'wsp_score' + ('' if n_tiers > 1 else '_' + tiers[0]))
        if show:
            plt.show()
    return fig
