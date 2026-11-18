"""Runs the primary LLM's program, once, on the Linux side in the sandbox venv.

Where it sits in the flow: step 4 of the session, oneshot_code tier. kernel.py starts this
file as a child process:
    <sandbox python> -I kernel_runner.py <work_dir>

Reads <work_dir>/code.py and kernel_args.json (written by kernel.py) and the tiny model's
files. Writes <work_dir>/output.txt and result.json, and the program's saved edits to the
session's edited/ folder.

Only the standard library and the sandbox venv's packages, plus edits.py, which is loaded by
file path because `-I` keeps this file's own folder off sys.path. Everything Linux-only or heavy
is imported inside functions, so prompts.py and kernel.py can import this module on Windows to
read PRELUDE_DOC and the file names.
"""
import importlib.util
import json
import os
import signal
import sys
import time
import traceback

ARGS_FILE = 'kernel_args.json'
CODE_FILE = 'code.py'
OUTPUT_FILE = 'output.txt'
RESULT_FILE = 'result.json'
STDOUT_TEMP = 'stdout.tmp'
STDERR_TEMP = 'stderr.tmp'

# What the prompt says is defined in advance. It lives next to the namespace that defines those
# names (build_prelude), so the prompt cannot describe a helper the runner does not have.
PRELUDE_DOC = """  NAMES                  the editable tensor names (the list above)
  SHAPES                 dict name -> shape
  load(name)             the tensor as a float32 torch tensor (original values)
  save_edit(name, t)     replace that tensor with t; refused if the name is not in NAMES, the shape differs, any value is not finite, or the edit only rescales the tensor
  drop_edit(name)        undo a save_edit
  edited()               names saved so far
  tok                    the tokenizer: tok.encode(text) -> ids, tok.decode(ids) -> text"""


class KernelTimeout(BaseException):
    """The time limit. Not an Exception, so a program's `except Exception` cannot swallow it."""


class CappedWriter:
    """A text stream that passes the first `max_chars` characters to a file and drops the rest,
    so a program printing in a loop cannot fill the disk."""

    def __init__(self, path, max_chars):
        self.file = open(path, 'w', encoding='utf-8', errors='replace')
        self.remaining = max_chars

    def write(self, text):
        piece = text[:self.remaining]
        if piece:
            self.file.write(piece)
            self.file.flush()  # flushed every time: a hard kill must not lose what was printed
            self.remaining -= len(piece)
        return len(text)

    def flush(self):
        self.file.flush()

    def fileno(self):
        return self.file.fileno()

    def isatty(self):
        return False


class TokenizerWrapper:
    """The prelude's `tok`: encode returns a plain list of ids, without special tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def assemble_output(work_dir, max_chars):
    """stdout then stderr, cut at max_chars, written to output.txt. The temporary files go away.
    Also called by kernel.py when the runner was killed before it could do this itself."""
    pieces = []
    for temp_name in (STDOUT_TEMP, STDERR_TEMP):
        temp_path = os.path.join(work_dir, temp_name)
        if os.path.exists(temp_path):
            with open(temp_path, encoding='utf-8', errors='replace') as f:
                pieces.append(f.read(max_chars + 1))
            os.remove(temp_path)
    output = ''.join(pieces)
    if len(output) > max_chars:
        output = output[:max_chars] + f'\n[output cut at {max_chars:,} characters]'
    with open(os.path.join(work_dir, OUTPUT_FILE), 'w', encoding='utf-8') as f:
        f.write(output)
    return output


def set_limits(args):
    import resource
    import torch
    resource.setrlimit(resource.RLIMIT_AS, (args['mem_bytes'], args['mem_bytes']))
    torch.set_grad_enabled(False)
    torch.set_num_threads(args['threads'])


def load_edits_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edits.py')
    spec = importlib.util.spec_from_file_location('edits', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prelude(args):
    """The program's global namespace: everything in PRELUDE_DOC, plus load_original (an alias)."""
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer

    edits = load_edits_module()
    edited_dir = args['edited_dir']
    names, frozen, readable = args['names'], args['frozen'], set(args['language_names'])

    def load(name):
        if name not in readable:
            raise KeyError(f"'{name}' is not a tensor of the language model")
        with safe_open(args['weights_path'], framework='pt') as f:
            return f.get_tensor(name).to(torch.float32, copy=True)

    def save_edit(name, t):
        original = load(name) if name in names else None
        try:
            return edits.save_edit(edited_dir, name, t, original, names, frozen)
        except edits.EditRefused as exc:
            # Printed as well as raised: a program may catch the refusal, and a later reader of
            # the output still needs to see that it happened.
            print(f'[save_edit refused] {exc}', file=sys.stderr)
            raise

    tokenizer = Tokenizer.from_file(args['tokenizer_path'])
    return {'__name__': '__main__',
            'NAMES': list(names),
            'SHAPES': {name: tuple(args['shapes'][name]) for name in names},
            'load': load,
            'load_original': load,
            'save_edit': save_edit,
            'drop_edit': lambda name: edits.drop_edit(edited_dir, name),
            'edited': lambda: edits.edited(edited_dir),
            'tok': TokenizerWrapper(tokenizer)}


def _raise_timeout(signum, frame):
    raise KernelTimeout()


def run_program(code, namespace, timeout_s):
    """(ok, exception text or None). Exceptions and the time limit are outcomes, not failures."""
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_s)
    try:
        exec(compile(code, CODE_FILE, 'exec'), namespace)
        return True, None
    except KernelTimeout:
        return False, f'timed out after {timeout_s} s'
    except SystemExit as exc:
        if exc.code in (None, 0):
            return True, None
        return False, f'SystemExit: {exc.code}'
    except BaseException as exc:
        traceback.print_exc()
        return False, f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)


def redirect_output(work_dir, max_chars):
    """Send stdout and stderr to capped files, at both levels: the Python streams, and the file
    descriptors that C libraries write to directly. One character over the cap gets through, so
    assemble_output can tell that the output was cut and say so."""
    stdout = CappedWriter(os.path.join(work_dir, STDOUT_TEMP), max_chars + 1)
    stderr = CappedWriter(os.path.join(work_dir, STDERR_TEMP), max_chars + 1)
    os.dup2(stdout.fileno(), 1)
    os.dup2(stderr.fileno(), 2)
    sys.stdout, sys.stderr = stdout, stderr
    return stdout, stderr


def main(work_dir):
    os.chdir(work_dir)
    with open(os.path.join(work_dir, ARGS_FILE), encoding='utf-8') as f:
        args = json.load(f)
    with open(os.path.join(work_dir, CODE_FILE), encoding='utf-8') as f:
        code = f.read()
    edits = load_edits_module()

    stdout, stderr = redirect_output(work_dir, args['max_output_chars'])
    set_limits(args)
    namespace = build_prelude(args)

    started = time.time()
    ok, exception = run_program(code, namespace, args['timeout_s'])
    seconds = round(time.time() - started, 2)

    stdout.flush()
    stderr.flush()
    assemble_output(work_dir, args['max_output_chars'])
    result = {'ok': ok, 'seconds': seconds, 'exception': exception, 'edited': edits.edited(args['edited_dir'])}
    temporary = os.path.join(work_dir, RESULT_FILE + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(result, f)
    os.replace(temporary, os.path.join(work_dir, RESULT_FILE))
    os._exit(0)  # threads the program left running must not keep the process alive


if __name__ == '__main__':
    main(sys.argv[1])
