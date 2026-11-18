"""Run a small causal language model with plain transformers, one prompt at a time.

Where it sits in the flow: the engine for a tiny model that vLLM cannot load (TinyStories-33M:
GPT-Neo is not in vLLM 0.29.0's registry, and vLLM refuses its transformers implementation).
tools/run_practice_baseline.py uses it for the practice baseline and tools/serve_transformers.py
serves it for scoring, so both run the model the same way. It runs on the Linux side in the vLLM
venv (which has torch and transformers), on the GPU if there is one.

One prompt at a time, never a padded batch: these models have learned position embeddings, and
padding would shift what position a question's tokens sit at. A 68M-parameter model answers a
16-token question in well under a second, so there is nothing to gain from batching.

Reads the model folder. Writes nothing. torch and transformers are imported inside functions.
"""
FINISH_STOP, FINISH_CUT = 'stop', 'length'


def load(model_dir, dtype='bfloat16'):
    """(model, tokenizer, version string), the model on the GPU if there is one, in eval mode."""
    import torch
    import transformers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
    return model.to(device).eval(), tokenizer, f'transformers {transformers.__version__}'


def generate(model, tokenizer, prompt, sampling, stop_ids, seed=None):
    """(text, number of new tokens, finish reason) for one prompt.

    The prompt is tokenised as written, with no special tokens added. sampling: the protocol's
    dict (temperature, max_tokens, and optionally top_k, top_p, repetition_penalty); temperature
    0 is greedy decoding. Generation stops at a token of stop_ids ('stop') or at max_tokens
    ('length'); the stop token is not part of the text.
    """
    import torch
    input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
    temperature = sampling.get('temperature', 1.0)
    arguments = {'max_new_tokens': sampling['max_tokens'], 'eos_token_id': list(stop_ids),
                 'pad_token_id': stop_ids[0], 'do_sample': temperature > 0}
    if temperature > 0:
        # A protocol that names no top_k or top_p means none: 0 and 1.0 switch them off explicitly,
        # whatever defaults the model's own generation config carries.
        arguments.update(temperature=temperature, top_k=sampling.get('top_k') or 0, top_p=sampling.get('top_p') or 1.0)
        if seed is not None:
            torch.manual_seed(seed)
    if sampling.get('repetition_penalty') is not None:
        arguments['repetition_penalty'] = sampling['repetition_penalty']
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **arguments)
    new_ids = output[0, input_ids.shape[1]:].tolist()
    stopped = bool(new_ids) and new_ids[-1] in stop_ids
    text = tokenizer.decode(new_ids[:-1] if stopped else new_ids, skip_special_tokens=True)
    return text, len(new_ids), FINISH_STOP if stopped else FINISH_CUT


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)
