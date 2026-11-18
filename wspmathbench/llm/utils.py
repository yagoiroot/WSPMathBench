"""Small shared helpers: JSON encoding, error reporting, the live status footer, CSV writing and
answer parsing.

Where it sits in the flow: below everything else in the package except config. It imports no other
module of the package but config, so query.py, registry.py and the benchmark code can all use it.

Writes one error log per process under cfg.ERROR_LOG_DIR, and only when an error is reported.
"""
import asyncio
import contextlib
import contextvars
import datetime
import json
import math
import os
import re
import socket
import threading
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from wspmathbench import config as cfg

# Serialises the read-append-write of a master CSV among the tasks of one process.
_save_lock = asyncio.Lock()

_MACHINE = re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname()) or 'unknown'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



# ------------------------------------------------------------------------------ error reporting
# A one-line summary for the console (type, message, and where it was raised), with the full
# traceback appended to this run's log file instead. One log per process and machine:
# <cfg.ERROR_LOG_DIR>/errors_<machine>_<start time>.log.
_ERROR_LOG_PATH = os.path.join(
    cfg.ERROR_LOG_DIR, f"errors_{_MACHINE}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log")
_EXC_MESSAGE_LIMIT = 160  # characters of the exception message kept on the console line

def _short_path(filename):
    # library frames read as e.g. httpx/_transports/default.py; project frames as the basename
    parts = os.path.normpath(filename).split(os.sep)
    for anchor in ('site-packages', 'Lib'):
        if anchor in parts:
            return '/'.join(parts[len(parts) - parts[::-1].index(anchor):])
    return os.path.basename(filename)

def _in_repository(filename):
    try:
        return os.path.commonpath([os.path.abspath(filename), cfg.REPO_ROOT]) == cfg.REPO_ROOT
    except ValueError:  # another drive, or a pseudo-file such as <string>
        return False

def exc_location(exc):
    """' @ <file>:<line> in <function>' for the innermost frame in this repository's own files,
    plus ' (raised in <library file>:<line>)' when the exception came from deeper, inside a
    library. Empty string if the exception carries no traceback."""
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if _in_repository(f.filename)]
    where = ''
    if ours:
        where += f' @ {os.path.basename(ours[-1].filename)}:{ours[-1].lineno} in {ours[-1].name}'
    if frames and (not ours or frames[-1] is not ours[-1]):
        where += f' (raised in {_short_path(frames[-1].filename)}:{frames[-1].lineno})'
    return where

def describe_exc(exc):
    """One-line summary: "<Type>: <message> @ <file>:<line> in <function>".
    e.g. "KeyError: 'cost_estimate' @ session.py:212 in run_session".
    A wrapped exception (raise ... from cause) also names the cause when the cause
    was raised somewhere else in this repository."""
    message = ' '.join(str(exc).split())
    if len(message) > _EXC_MESSAGE_LIMIT:
        message = message[:_EXC_MESSAGE_LIMIT] + '...'
    text = f'{type(exc).__name__}: {message}{exc_location(exc)}'
    cause = exc.__cause__
    if cause is not None and ' @ ' in exc_location(cause) and exc_location(cause) != exc_location(exc):
        text += f' <- caused by {describe_exc(cause)}'
    return text

def log_exc(exc, context=''):
    """Append exc's full traceback, headed by a timestamp and `context`, to this run's error
    log. Returns the log's path, relative to the repository where it can be, or None if it could
    not be written (logging must never be what breaks a run)."""
    try:
        os.makedirs(cfg.ERROR_LOG_DIR, exist_ok=True)
        with open(_ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f'===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {context}\n')
            f.write(''.join(traceback.format_exception(exc)))
            f.write('\n')
    except OSError:
        return None
    try:
        return os.path.relpath(_ERROR_LOG_PATH, cfg.REPO_ROOT)
    except ValueError:  # the results folder is on another drive
        return _ERROR_LOG_PATH

def report_exc(exc, context=''):
    """describe_exc() for the console, with the full traceback logged via log_exc(). Connection
    errors, timeouts and interrupted streams are routine and frequent, so they are not logged."""
    message = ' '.join(str(exc).split())
    if type(exc).__name__ in ['APIConnectionError', 'TimeoutError', 'StreamInterruptedError']:
        return describe_exc(exc)
    if message == 'Upstream idle timeout exceeded':
        return describe_exc(exc)
    path = log_exc(exc, context)
    return describe_exc(exc) + (f' (full trace: {path})' if path else '')


# --------------------------------------------------------------------------- live status footer
# Display only. While a run is in progress, a footer under the normal prints shows what each of its
# entries (a question of a scoring pass, or an edit session) is doing, for example:
#     [GSM8K_500 12/500 done | 3 active | ~tokens]  i3 ~1840  i7 r2 ~212  i9 14s
# Streamed characters become approximate tokens (chars // 4) or raw character counts; an entry
# with no streamed output yet shows its elapsed time. How the pieces fit:
#   1. live_status(name, total, mode) opens a run's footer and yields its status object.
#   2. live_status_task(status, i, coro) runs one entry as index i, as its own asyncio task.
#   3. live_status_attempt(attempt) marks the start of each attempt: the timer and count restart.
#   4. live_status_tick(n_chars) is called by the stream collectors in query.py per stream delta.
# Rich redraws up to LIVE_STATUS_MAX_LINES lines per run and erases the footer when the last run
# finishes.

LIVE_STATUS_DEFAULT = False  # what live_status=None falls back to: False, 'tokens' or 'chars'

LIVE_STATUS_MAX_LINES = 5    # footer height per run, summary included

_LIVE_STATUS_CHARS_PER_TOKEN = 4  # rough divisor turning streamed characters into '~tokens'

# The (run status, question index) belonging to the current asyncio task and its children.
_live_status_index = contextvars.ContextVar('live_status_index', default=None)

_live_display = None         # the one rich Live of this process, shared by concurrent runs

_live_runs = []              # status objects currently on the footer

_live_runs_lock = threading.Lock()  # guards _live_runs, which the render thread also reads

@contextlib.contextmanager
def live_status(name, total, mode=None):
    """Open one run's live footer, yielding its status or None when disabled/unsupported.

    mode: None -> LIVE_STATUS_DEFAULT; False -> off; True/'tokens' -> ~tokens; 'chars' -> chars.
    The first active run starts the process's shared Rich Live; the last one stops it. Normal
    stdout/stderr prints scroll above the footer. All implementation helpers are nested here so
    the footer lifecycle and rendering logic can be read together.
    """
    global _live_display
    mode = LIVE_STATUS_DEFAULT if mode is None else mode
    if mode is True: mode = 'tokens'
    if not mode:
        yield None
        return
    if mode not in ('tokens', 'chars'):
        raise ValueError(f"live_status must be False, True, 'tokens' or 'chars'; got {mode!r}")

    class LiveStatus:
        """One run's scoreboard; worker tasks write it and Rich's render thread reads it."""
        def __init__(self):
            self.name, self.total, self.mode = name, total, mode
            self.done = 0
            # question index -> [attempt start, 1-based attempt number, streamed characters]
            self.active = {}
            self.lock = threading.Lock()

        def attempt(self, i, attempt):
            """Start/restart an index's zero-based attempt, resetting its timer and count."""
            with self.lock:
                self.active[i] = [time.monotonic(), attempt + 1, 0]

        def tick(self, i, n_chars):
            """Add streamed characters to an active index; ignore indices that are not active."""
            with self.lock:
                entry = self.active.get(i)
                if entry is not None: entry[2] += n_chars

        def finish(self, i):
            """Remove an index from the footer and count its question as done."""
            with self.lock:
                self.active.pop(i, None)
                self.done += 1

        def render(self, width):
            """Format this run as Rich Text, including only entries that fit the line budget."""
            def format_elapsed(seconds):
                seconds = int(seconds)
                return f'{seconds}s' if seconds < 60 else f'{seconds // 60}m{seconds % 60:02d}s'

            def format_entry(i, started, attempt, chars):
                retry = f'r{attempt} ' if attempt > 1 else ''
                if chars == 0:
                    progress = format_elapsed(time.monotonic() - started)
                elif self.mode == 'tokens':
                    progress = f'~{chars // _LIVE_STATUS_CHARS_PER_TOKEN}'
                else:
                    progress = f'{chars}c'
                return f'i{i} {retry}{progress}'

            def pack_lines(lines, entries):
                for k, entry in enumerate(entries):
                    if len(lines[-1]) + 2 + len(entry) <= width:
                        lines[-1] += '  ' + entry
                    elif len(lines) < LIVE_STATUS_MAX_LINES:
                        lines.append('  ' + entry)
                    else:  # trim the last line until its overflow marker fits
                        hidden = len(entries) - k
                        while (len(lines[-1]) + len(f'  ... +{hidden} more') > width
                               and '  ' in lines[-1].strip()):
                            lines[-1] = lines[-1].rsplit('  ', 1)[0]
                            hidden += 1
                        lines[-1] += f'  ... +{hidden} more'
                        break
                return lines

            with self.lock:  # copy under the lock; do not block workers during formatting
                snapshot = sorted(self.active.items())
                done = self.done
            unit = '~tokens' if self.mode == 'tokens' else 'chars'
            lines = [f'[{self.name} {done}/{self.total} done | {len(snapshot)} active | {unit}]']
            entries = [format_entry(i, *entry) for i, entry in snapshot]
            return Text('\n'.join(pack_lines(lines, entries)), no_wrap=True, overflow='ellipsis')

    console = _live_display.console if _live_display is not None else Console()

    def render_all_runs():
        """Build the shared footer; rendering failures must never interrupt a benchmark."""
        try:
            with _live_runs_lock:
                runs = list(_live_runs)
            return Group(*(run.render(console.width) for run in runs))
        except Exception:
            return Text('')

    if _live_display is None:
        if not console.is_terminal or console.is_dumb_terminal:
            print('[live status] this console cannot redraw in place, continuing without the '
                  'status footer (in PyCharm, enable "Emulate terminal in output console")')
            yield None
            return
        _live_display = Live(get_renderable=render_all_runs, console=console, refresh_per_second=2,
                             transient=True, redirect_stdout=True, redirect_stderr=True)
        _live_display.start()
    status = LiveStatus()
    with _live_runs_lock:
        _live_runs.append(status)
    try:
        yield status
    finally:
        with _live_runs_lock:
            _live_runs.remove(status)
            last = not _live_runs
        if last:
            _live_display.stop()
            _live_display = None

async def live_status_task(status, i, coro):
    """Run one worker coroutine as index i of `status` (a no-op wrapper when None).

    Step 2 in the overview above. It stores (status, i) in the ContextVar, so every
    live_status_attempt and live_status_tick made under this worker credits index i. It must
    run as its own asyncio task (as asyncio.gather does), so that the value stays private to this
    worker. When the worker ends, normally or not, the index is marked finished."""
    if status is None:
        return await coro
    _live_status_index.set((status, i))  # this task's own context, inherited by its children
    try:
        return await coro
    finally:
        status.finish(i)

def live_status_attempt(attempt):
    """Mark the start of a worker attempt (0-based) for the current index: resets its timer and count.

    Step 3 in the overview above. Does nothing outside a footer-enabled run."""
    current = _live_status_index.get()  # (LiveStatus, question index), or None
    if current is not None:
        current[0].attempt(current[1], attempt)

def live_status_tick(n_chars):
    """Credit n_chars of streamed text to the current index.

    Step 4 in the overview above. The stream collectors in query.py call this with len(delta).
    The ContextVar supplies which run and index to credit, so the collectors do not need to know
    what is running them. Does nothing outside a footer-enabled run."""
    current = _live_status_index.get()  # (LiveStatus, question index), or None
    if current is not None:
        current[0].tick(current[1], n_chars)


# --------------------------------------------------------------------------------- small helpers

def data_to_text_data_format(data, location):
    """Write a data structure to a text file as its own string representation; read it back with
    text_to_data_format. Works for any structure whose str() is a Python literal."""
    with open(location, 'w', encoding='utf-8') as f:
        f.write(str(data))
    print(f'file {location} created successfully')

def text_to_data_format(location):
    """Read back a file written by data_to_text_data_format. nan is accepted as a value."""
    with open(location, encoding='utf-8') as f:
        text = f.read()
    return eval(text, {"__builtins__": {}}, {"nan": math.nan, "NaN": math.nan})

def ladder(name, n, lo=0.10, hi=0.85):
    """n colours evenly spaced between lo and hi of a colormap, given by name or as an object."""
    if type(name) == str:
        return [plt.get_cmap(name)(v) for v in np.linspace(lo, hi, n)]
    else:
        return [name(v) for v in np.linspace(lo, hi, n)]

def write_csv_with_retry(df, path, attempts=6, base_delay=0.25):
    """df.write_csv(path), retried with a growing wait while another program (a file-sync client,
    a spreadsheet) briefly holds the file open."""
    for i in range(attempts):
        try:
            df.write_csv(path)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)

def wait_strat2_helper(x):
    """Seconds to wait before retry number x under retry_wait='strat2': rises from about 2-5 s
    towards 7-10 s, with jitter so that concurrent workers do not retry in step."""
    return (5 / (np.pi / 2)) * np.arctan(x) + 1 +np.random.uniform(1,4)

def last_boxed_only_string(string):
    """The last \\boxed{...} (or \\fbox{...}) of a string, braces included, or None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, depth, right = idx, 0, None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return None if right is None else string[idx:right + 1]

def remove_boxed(s):
    """The content of a string that is exactly one \\boxed{...}, or None."""
    left = "\\boxed{"
    if s and s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return None

def safer_int(x):
    """int() that also reads strings such as '44.00', '1,600' and '50%'. Anything else raises."""
    if type(x) == int:
        return x
    x=x.replace(',','')
    x=x.replace('%','')
    x = x.replace("\\", '')
    return int(float(x))
