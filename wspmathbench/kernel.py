"""oneshot_code: run the primary LLM's program in the sandbox venv and read back what it did.

Where it sits in the flow: step 4 of the session, oneshot_code tier. The program runs on the Linux
side (inside WSL on Windows, see cfg.shell_command) as a separate process, in a separate venv,
with a minimal environment, a memory limit and a time limit. That is NOT isolation: the program
runs as the Linux user, can read and write everything that user can (under WSL that includes the
Windows drives under /mnt) and has the network. Run sessions of this tier only on a machine where
that is acceptable.

Writes <work_dir>/code.py and kernel_args.json. Reads back output.txt and result.json, which
kernel_runner.py writes on the Linux side.
"""
import json
import os
import shlex
import subprocess
import time

from wspmathbench import config as cfg
from wspmathbench import edits
from wspmathbench import kernel_runner as runner

_DIAGNOSTIC_CHARS = 1000


def run(code, work_dir, tiny, limits=None):
    """Run `code` once. Returns {'ok', 'seconds', 'exception', 'edited', 'out'}.

    An exception or a timeout inside the program is recorded here (ok False), never retried.
    limits: keys of cfg.SANDBOX to override; only the selftest uses it, with short limits.
    """
    limits = {**cfg.SANDBOX, **(limits or {})}
    if limits['mode'] == 'bwrap':
        raise NotImplementedError('v2')
    if limits['mode'] != 'subprocess':
        raise ValueError(f"Unknown sandbox mode {limits['mode']!r}.")

    _write_inputs(code, work_dir, tiny, limits)
    started = time.time()
    failure_reason = _run_kernel(work_dir, limits)
    seconds = round(time.time() - started, 2)
    return _read_back(work_dir, limits, seconds, failure_reason)


def _write_inputs(code, work_dir, tiny, limits):
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, runner.CODE_FILE), 'w', encoding='utf-8') as f:
        f.write(code)
    args = {'weights_path': cfg.to_wsl_path(tiny.weights_path),
            'tokenizer_path': cfg.to_wsl_path(os.path.join(tiny.model_dir, cfg.MODEL_TOKENIZER_FILE)),
            'edited_dir': cfg.to_wsl_path(_edited_dir(work_dir)),
            'names': tiny.names, 'language_names': tiny.language_names,
            'shapes': tiny.shapes, 'frozen': tiny.frozen,
            'timeout_s': limits['timeout_s'], 'mem_bytes': limits['mem_bytes'],
            'threads': limits['threads'], 'max_output_chars': limits['max_output_chars']}
    with open(os.path.join(work_dir, runner.ARGS_FILE), 'w', encoding='utf-8') as f:
        json.dump(args, f)


def _edited_dir(work_dir):
    # The program's edits go next to its work folder, not inside it.
    return os.path.normpath(os.path.join(work_dir, '..', cfg.SESSION_EDITED_DIR))


def _bash_command(work_dir, limits):
    # `python` is left unquoted so bash expands its ~; the other paths may contain spaces.
    # `timeout` runs on the Linux side because, under WSL, killing wsl.exe does not reliably stop
    # the child.
    work_dir_wsl = shlex.quote(cfg.to_wsl_path(work_dir))
    runner_path = shlex.quote(cfg.to_wsl_path(cfg.KERNEL_RUNNER_FILE))
    hard_limit = limits['timeout_s'] + limits['hard_kill_margin_s']
    return (f"exec env -i HOME=$HOME PATH=/usr/bin:/bin OMP_NUM_THREADS={limits['threads']} "
            f"CUDA_VISIBLE_DEVICES= timeout --signal=KILL {hard_limit} "
            f"{limits['python']} -I {runner_path} {work_dir_wsl}")


def _run_kernel(work_dir, limits):
    """Run the runner. Returns None, or a sentence if the process itself did not end normally."""
    command = cfg.shell_command(_bash_command(work_dir, limits))
    try:
        completed = subprocess.run(command, capture_output=True,
                                   timeout=limits['timeout_s'] + limits['wsl_margin_s'])
    except subprocess.TimeoutExpired:
        return f"the kernel process did not end within {limits['timeout_s'] + limits['wsl_margin_s']} s"
    if completed.returncode == 0:
        return None
    return (f'the kernel process ended with exit code {completed.returncode} (9 or 137 mean it was '
            f'killed, by the hard time limit or for running out of memory): '
            f'{_diagnostic_text(completed.stderr)[-_DIAGNOSTIC_CHARS:]}')


def _diagnostic_text(raw):
    # wsl.exe's own messages are UTF-16 and the Linux side's are UTF-8; dropping the NUL bytes
    # keeps either readable. wsl.exe's lines ("wsl: Failed to mount G:\...") are about the machine,
    # not the program, so they are left out.
    text = raw.decode('utf-8', errors='replace').replace('\x00', '')
    return '\n'.join(line for line in text.splitlines() if not line.startswith('wsl:')).strip()


def _read_back(work_dir, limits, seconds, failure_reason):
    result_path = os.path.join(work_dir, runner.RESULT_FILE)
    if os.path.exists(result_path):
        with open(result_path, encoding='utf-8') as f:
            result = json.load(f)
        with open(os.path.join(work_dir, runner.OUTPUT_FILE), encoding='utf-8') as f:
            result['out'] = f.read()
        return result

    # The runner never got to finish. Whatever it saved and printed before that still counts.
    reason = failure_reason or f'the kernel process wrote no {runner.RESULT_FILE}'
    return {'ok': False, 'seconds': seconds, 'exception': reason,
            'edited': edits.edited(_edited_dir(work_dir)),
            'out': runner.assemble_output(work_dir, limits['max_output_chars'])}
