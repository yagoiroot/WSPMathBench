"""A wrapper around math-verify that cannot hang and does not fail silently.

Where it sits in the flow: benchmarks.eval_benchmark_async and analysis.py use it to decide whether a
MATH-500 answer equals the reference answer.

math-verify's own timeout cannot be used on Windows (its parse() then returns nothing, silently), and
without a timeout SymPy can run forever on an answer such as a huge binomial. So this file launches
ITSELF as a plain subprocess: the worker loop runs under `__main__`, and the parent kills and restarts
it when one answer takes more than `timeout` seconds.

Statuses: 'correct', 'incorrect', 'unparsed' (no parsable answer), 'timeout', 'error'. An unparsable
reference answer or a failed startup check raises, because those mean the grader is broken, not that
the model is wrong.

Reads and writes no files. The worker needs the math-verify package.
"""
import json
import os
import queue
import subprocess
import sys
import threading


class MathVerifyGrader:
    def __init__(self, timeout=10.0, startup_timeout=120.0):
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self._proc = None
        self._start()

    def _start(self):
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8')
        self._replies = queue.Queue()  # one queue per process, so a killed worker cannot leave a stale reply
        threading.Thread(target=self._read, args=(self._proc, self._replies), daemon=True).start()
        try:
            ready = self._replies.get(timeout=self.startup_timeout)
        except queue.Empty:
            self._kill()
            raise RuntimeError('math_verify grader worker did not start in time')
        if ready is None or not ready.get('ready'):
            self._kill()
            raise RuntimeError(f'math_verify grader failed its startup check: {ready}')

    @staticmethod
    def _read(proc, replies):
        for line in proc.stdout:
            replies.put(json.loads(line))
        replies.put(None)  # EOF: the worker died

    def _kill(self):
        self._proc.kill()
        self._proc.wait()

    def grade(self, gold, model_answer):
        self._proc.stdin.write(json.dumps([gold, model_answer]) + '\n')
        self._proc.stdin.flush()
        try:
            reply = self._replies.get(timeout=self.timeout)
        except queue.Empty:
            self._kill()
            self._start()
            return 'timeout'
        if reply is None:  # worker crashed on this answer (e.g. out of memory)
            self._kill()
            self._start()
            return 'error'
        if reply['status'] == 'gold_unparsed':
            raise ValueError(f'math_verify could not parse the gold answer {gold!r}')
        return reply['status']

    def close(self):
        self._proc.stdin.close()
        try:
            self._proc.wait(5)
        except subprocess.TimeoutExpired:
            self._kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker():
    protocol_out, sys.stdout = sys.stdout, sys.stderr  # keep stray prints out of the reply stream
    import logging
    logging.disable(logging.WARNING)  # the disabled-timeout notices: this process's parent enforces the timeout
    from math_verify import parse, verify

    def grade(gold, model_answer):
        try:
            parsed_gold = parse(f'${gold}$', parsing_timeout=None, raise_on_error=True)
        except Exception:
            return 'gold_unparsed'
        # parse() falls back to [raw_string] or [] when no expression was found; a usable gold has a parsed value
        if len(parsed_gold) < 2 and not any(not isinstance(p, str) for p in parsed_gold):
            return 'gold_unparsed'
        predicted = parse(f'${model_answer}$', parsing_timeout=None, raise_on_error=True)
        if not predicted:
            return 'unparsed'
        return 'correct' if verify(parsed_gold, predicted, timeout_seconds=None, raise_on_error=True) else 'incorrect'

    def send(obj):
        protocol_out.write(json.dumps(obj) + '\n')
        protocol_out.flush()

    canary = (grade('9', '9'), grade(r'\frac{1}{2}', '0.5'), grade('9', '8'))
    if canary != ('correct', 'correct', 'incorrect'):
        send({'ready': False, 'canary': canary})
        return
    send({'ready': True})

    for line in sys.stdin:
        gold, model_answer = json.loads(line)
        try:
            send({'status': grade(gold, model_answer)})
        except Exception as exc:
            send({'status': 'error', 'detail': repr(exc)})


if __name__ == '__main__':
    _worker()
