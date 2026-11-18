"""Canned replies for provider='stub', so the whole benchmark runs with no API spend.

Where it sits in the flow: step 4 of the session calls `query_stub` in place of
registry.query_base_model. The stub is deliberately not in BASE_MODEL_REGISTRY, which lists real
providers only. The replies are test fixtures built from the TinyModel (`targets`), so they are valid
for any registered tiny model; the selftest computes what it expects from the same function.

Reads and writes nothing.
"""
import json

RETRY_MARKER = 'PREVIOUS ATTEMPT (not applied)'  # the heading of prompts.RETRY_SUFFIX
STUB_MODELS = ('stub-spec', 'stub-spec-invalid-then-valid', 'stub-code', 'stub-code-raises', 'stub-no-block')
TOKENS = ('the', ' the')  # ordinary words, as in the edit language's own examples
ZEROED_COLUMNS = (0, 1, 2, 3)
SCALED_COLUMNS, RAISING_SCALED_COLUMNS = 16, 8


def targets(tiny):
    """The two tensors the stub edits: the MLP output of the middle layer and of the last layer."""
    mlp_outputs = [name for name in tiny.residual_writers()
                   if name.endswith(tiny.entry['residual_writer_suffixes'][0])]
    return mlp_outputs[len(mlp_outputs) // 2], mlp_outputs[-1]


def missing_tensor(tiny):
    """A well-formed tensor name in a layer the model does not have."""
    suffix = tiny.entry['residual_writer_suffixes'][0]
    return tiny.entry['layer_stem'].format(layer=len(tiny.layer_types) + 6) + suffix


def out_of_range_column(tiny):
    first, _ = targets(tiny)
    return tiny.shapes[first][1] + 416


def _fenced(kind, body):
    return f'```{kind}\n{body}\n```\n'


def _valid_spec_reply(tiny):
    first, second = targets(tiny)
    spec = {'edits': [
        {'op': 'project_out', 'tensor': first,
         'direction': {'type': 'unembed_mean', 'tokens': list(TOKENS), 'fold_final_norm': True},
         'strength': 1.0, 'cols': 'all'},
        {'op': 'zero_columns', 'tensor': second, 'cols': list(ZEROED_COLUMNS)}]}
    return 'The stub reasons in a sentence, then gives its one block.\n\n' + _fenced('json', json.dumps(spec, indent=1))


def _invalid_spec_reply(tiny):
    first, _ = targets(tiny)
    spec = {'edits': [
        {'op': 'zero_columns', 'tensor': first, 'cols': [out_of_range_column(tiny)]},
        {'op': 'zero_rows', 'tensor': missing_tensor(tiny), 'rows': [0]}]}
    return 'The stub makes two mistakes on its first attempt.\n\n' + _fenced('json', json.dumps(spec, indent=1))


def _code_reply(tiny):
    first, _ = targets(tiny)
    return f'The stub scales {SCALED_COLUMNS} columns of one MLP output.\n\n' + _fenced('python', '\n'.join([
        f'name = {first!r}',
        'original = load(name)',
        'w = original.clone()',
        f'w[:, :{SCALED_COLUMNS}] *= 0.5',
        "print('relative change', float((w - original).norm() / original.norm()))",
        'save_edit(name, w)',
        "print('edited:', edited())"]))


def _raising_code_reply(tiny):
    _, second = targets(tiny)
    return 'The stub saves one edit, then its program fails.\n\n' + _fenced('python', '\n'.join([
        f'name = {second!r}',
        'w = load(name)',
        f'w[:, :{RAISING_SCALED_COLUMNS}] *= 0.25',
        'save_edit(name, w)',
        "print('saved one edit, about to raise')",
        "raise RuntimeError('deliberate stub failure after one save_edit')"]))


_NO_BLOCK_REPLY = 'The stub only talks, in prose, and never gives a block.'


def _reply_text(model, prompt, tiny):
    if model == 'stub-spec':
        return _valid_spec_reply(tiny)
    if model == 'stub-spec-invalid-then-valid':
        return _valid_spec_reply(tiny) if RETRY_MARKER in prompt else _invalid_spec_reply(tiny)
    if model == 'stub-code':
        return _code_reply(tiny)
    if model == 'stub-code-raises':
        return _raising_code_reply(tiny)
    if model == 'stub-no-block':
        return _NO_BLOCK_REPLY
    raise ValueError(f"Unknown stub model {model!r}; the stub models are {STUB_MODELS}.")


async def query_stub(model, prompt, tiny):
    """A standard-contract response dict with zero tokens and zero cost."""
    return {'text': _reply_text(model, prompt, tiny), 'reasoning_text': None, 'prompt_tokens': 0,
            'cached_tokens': 0, 'cache_write_tokens': 0, 'completion_tokens': 0,
            'reasoning_tokens': 0, 'total_tokens': 0, 'cost_estimate': 0.0, 'response': None,
            'status': 'completed', 'incomplete_reason': None, 'server_provider': 'stub'}
