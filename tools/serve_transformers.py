"""A minimal OpenAI-compatible completions server for a tiny model that vLLM cannot load.

Where it sits in the flow: scoring. The local-server code (llm/query.py: an entry of
LOCAL_MODEL_CONFIG with engine 'transformers', built by wsp_local_model) starts this on the Linux
side in place of `vllm serve`, and the scoring run then queries it like any other local model. By
hand, in a Linux shell at the repository root:

    ~/venvs/vllm/bin/python tools/serve_transformers.py --model-dir <models folder>/TinyStories-33M-bf16 \
        --served-model-name TinyStories-33M --port 8000

Endpoints: GET /health, GET /v1/models, POST /v1/completions (prompt, max_tokens, temperature,
top_p, top_k, repetition_penalty, seed). There is no chat endpoint: these models have no chat
template. Requests are answered one at a time (tools/transformers_engine.py says why), whatever
the number of client connections.

Reads the model folder. Writes nothing. Standard library plus transformers_engine.
"""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import run_practice_baseline as baseline
import transformers_engine

SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'repetition_penalty')


class Handler(BaseHTTPRequestHandler):
    server_version = 'wsp-transformers'

    def log_message(self, format, *args):  # noqa: A002 - the base class's signature
        pass  # one line per request would bury the console; errors are still printed below

    def do_GET(self):
        if self.path == '/health':
            return self._send(200, {'status': 'ok'})
        if self.path == '/v1/models':
            return self._send(200, {'object': 'list', 'data': [
                {'id': self.server.served_name, 'object': 'model', 'created': self.server.started, 'owned_by': 'wsp'}]})
        self._send(404, {'error': {'message': f'no such path: {self.path}'}})

    def do_POST(self):
        if self.path != '/v1/completions':
            return self._send(404, {'error': {'message': f'{self.path} is not served; only /v1/completions is '
                                                         f'(this model has no chat template)'}})
        try:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = request['prompt']
            if not isinstance(prompt, str):
                raise ValueError('`prompt` must be one string')
            sampling = {'max_tokens': int(request.get('max_tokens') or 16),
                        **{key: request[key] for key in SAMPLING_KEYS if request.get(key) is not None}}
            sampling.setdefault('temperature', 1.0)
            with self.server.generation_lock:
                text, n_new, finish_reason = transformers_engine.generate(
                    self.server.model, self.server.tokenizer, prompt, sampling, self.server.stop_ids, request.get('seed'))
            n_prompt = transformers_engine.count_tokens(self.server.tokenizer, prompt)
        except Exception as exc:  # a bad request or a failed generation: say so, keep serving
            print(f'request failed: {type(exc).__name__}: {exc}', flush=True)
            return self._send(400, {'error': {'message': f'{type(exc).__name__}: {exc}'}})
        self._send(200, {
            'id': 'cmpl-' + uuid.uuid4().hex, 'object': 'text_completion', 'created': int(time.time()),
            'model': self.server.served_name,
            'choices': [{'index': 0, 'text': text, 'finish_reason': finish_reason, 'logprobs': None}],
            'usage': {'prompt_tokens': n_prompt, 'completion_tokens': n_new, 'total_tokens': n_prompt + n_new}})

    def _send(self, status, body):
        data = json.dumps(body).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    model, tokenizer, version = transformers_engine.load(args.model_dir, args.dtype)
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server.model, server.tokenizer, server.served_name = model, tokenizer, args.served_model_name
    server.stop_ids = baseline.stop_token_ids(args.model_dir, None)
    server.generation_lock, server.started = threading.Lock(), int(time.time())
    print(f'{args.served_model_name}: {args.model_dir} with {version}, {args.dtype}, on {model.device}; '
          f'stop ids {server.stop_ids}; serving on port {args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
