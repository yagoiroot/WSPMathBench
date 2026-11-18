"""The oneshot_spec edit language: parse, validate, resolve directions, apply.

Where it sits in the flow: step 4 of the session, for the oneshot_spec tier only. The prompt's
description of the language is rendered here (render_dsl_text), from the same TinyModel the
validator reads, so the prompt and the validator cannot disagree about a model.

The language is one language. What varies with the tiny model is what the registry says about its
architecture: which layers have heads and how wide a head is, which tensors write to the residual
stream, how its final norm is folded in, and whether a norm sits between a tensor and the residual
stream. A row records that as dsl_version '<version>-<model label>' (cfg.version_label).

Errors that go back to the primary LLM are returned as lists of plain sentences, never raised.
Reads tensors through the TinyModel. `apply` writes <session>/edited/*.npy through edits;
nothing else here touches the disk.
"""
import json
import textwrap

import torch

from wspmathbench import config as cfg
from wspmathbench import edits

# The prompt's description of the language. Indented paragraphs are wrapped at _WRAP_WIDTH; the
# self-test compares the rendered prompts with reference copies, character for character. The
# example tokens are deliberately ordinary words: a token tied to a known failure of a tiny model
# would be a hint in a prompt that is meant to have none.
_WRAP_WIDTH = 86
_INDENT = '    '
_LIMITS_TEXT = '''Every direction is normalised to unit length. Limits: at most {max_ops} operations; at most
{max_indices} indices per operation ("all" is allowed only in project_out); factors and
strengths between 0 and 4; indices are zero-based.'''
_FOLD_FINAL_NORM_TEXT = {
    'rms': "the final norm's weight",
    'rms_one_plus': "the final norm's gain (1 + its weight)",
    'layernorm': "the final norm's weight (its centring and its bias are left out)",
}

OP_FIELDS = {
    'zero_head': ('layer', 'head'),
    'zero_columns': ('tensor', 'cols'),
    'zero_rows': ('tensor', 'rows'),
    'scale_columns': ('tensor', 'cols', 'factor'),
    'scale_rows': ('tensor', 'rows', 'factor'),
    'project_out': ('tensor', 'direction', 'strength', 'cols'),
}
# Required fields of each direction type, and the ones it may also carry. fold_post_norm is
# accepted only for a model that has post-block norms (_optional_direction_fields).
DIRECTION_FIELDS = {
    'unembed_token': (('token',), ('fold_final_norm',)),
    'unembed_mean': (('tokens',), ('fold_final_norm',)),
    'svd': (('tensor', 'component'), ()),
}
_UNEMBED_TYPES = ('unembed_token', 'unembed_mean')


def render_dsl_text(tiny):
    """The description of the edit language for this tiny model, as the prompt quotes it."""
    entry = tiny.entry
    head_dim = tiny.head_dim
    lines = [
        '{"op": "zero_head", "layer": L, "head": H}',
        _wrap(f"{_head_layers_text(tiny)}; zeroes head H's input columns of layer L's "
              f"{entry['attention_output_suffix'].lstrip('.')} (columns H*{head_dim} .. H*{head_dim}+{head_dim - 1})"),
        '{"op": "zero_columns", "tensor": NAME, "cols": [...]}',
        '{"op": "zero_rows", "tensor": NAME, "rows": [...]}',
        '{"op": "scale_columns", "tensor": NAME, "cols": [...], "factor": F}',
        '{"op": "scale_rows", "tensor": NAME, "rows": [...], "factor": F}',
        '{"op": "project_out", "tensor": NAME, "direction": DIR, "strength": S, "cols": [...] | "all"}',
        _wrap(f'only for tensors that write to the residual stream ({_writers_text(entry)}, shape '
              f'[{tiny.width}, k]); with unit direction u:'),
        _INDENT + 'W[:, cols] = W[:, cols] - S * u (u^T W[:, cols])',
    ]
    if entry['residual_bias_suffixes']:
        lines.append(_wrap(f'also for the bias vectors added with them ({_join_names(entry["residual_bias_suffixes"])}, '
                           f'shape [{tiny.width}]), with "cols": "all": b = b - S * u (u^T b)'))
    lines += [
        'Directions DIR, resolved without running the model:',
        '{"type": "unembed_token", "token": "the", "fold_final_norm": true}',
        _wrap('the output-embedding row of that token (it must encode to exactly one token id); with '
              f"fold_final_norm the row is multiplied elementwise by {_FOLD_FINAL_NORM_TEXT[entry['final_norm_kind']]}"),
        '{"type": "unembed_mean", "tokens": ["the", " the"], "fold_final_norm": true}',
        _INDENT + 'the mean of several such rows',
    ]
    if entry['post_norms']:
        lines.append(_wrap('in this model a norm with gain (1 + weight) sits between each residual-writing tensor '
                           'and the residual stream (' + _join_names(entry['post_norms'].values()) + '); an '
                           'unembed_token or unembed_mean direction is therefore also multiplied elementwise by '
                           'the gain of the norm that follows the edited tensor, so that what is removed is the '
                           'component that reaches the residual stream; "fold_post_norm": false in the direction '
                           'turns this off'))
    lines += [
        '{"type": "svd", "tensor": NAME, "component": K}',
        _INDENT + 'the K-th left singular vector of a residual-writing tensor',
        _LIMITS_TEXT.replace('{max_ops}', str(cfg.MAX_OPS)).replace('{max_indices}', str(cfg.MAX_INDICES)),
    ]
    return '\n'.join(lines)


def _head_layers_text(tiny):
    if len(tiny.head_layers()) == len(tiny.layer_types):
        return 'any layer'
    if tuple(tiny.entry['head_kinds']) != ('full_attention',):
        raise ValueError(f"No wording for layers with heads of kinds {tiny.entry['head_kinds']} ({tiny.key}).")
    return 'full-attention layers only'


def _wrap(paragraph):
    return textwrap.fill(paragraph, width=_WRAP_WIDTH, initial_indent=_INDENT, subsequent_indent=_INDENT,
                         break_long_words=False, break_on_hyphens=False)


def _join_names(suffixes):
    names = ['*' + suffix for suffix in suffixes]
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def _writers_text(entry):
    return _join_names(entry['residual_writer_suffixes'])


def _residual_writers_sentence(tiny):
    return f"it accepts only {_writers_text(tiny.entry)} tensors"


def parse(block):
    """(spec, errors) from the text of the ```json block. errors holds the decoder's message."""
    try:
        return json.loads(block), []
    except json.JSONDecodeError as exc:
        return None, [f'the block is not valid JSON: {exc}']


# ------------------------------------------------------------------------------- validation

def validate(spec, tiny):
    """Every problem in a parsed spec, one plain sentence each. An empty list means valid."""
    if not isinstance(spec, dict) or not isinstance(spec.get('edits'), list):
        return ['the block must be a JSON object of the form {"edits": [...]}']
    unknown = sorted(set(spec) - {'edits'})
    errors = [f"unknown top-level key '{key}'" for key in unknown]
    if len(spec['edits']) > cfg.MAX_OPS:
        errors.append(f"there are {len(spec['edits'])} operations, at most {cfg.MAX_OPS} are allowed")
    for i, op in enumerate(spec['edits']):
        errors += [f'edits[{i}]: {problem}' for problem in _validate_op(op, tiny)]
    return errors


def _validate_op(op, tiny):
    if not isinstance(op, dict):
        return ['an operation must be a JSON object']
    kind = op.get('op')
    if kind not in OP_FIELDS:
        return [f"unknown op {kind!r}; the ops are {', '.join(OP_FIELDS)}"]
    problems = _check_fields(op, OP_FIELDS[kind], (), f"op '{kind}'", ignore=('op',))
    if problems:
        return problems
    if kind == 'zero_head':
        return _validate_zero_head(op, tiny)
    problems = _validate_editable_tensor(op['tensor'], tiny)
    if problems:
        return problems
    if kind == 'project_out':
        return _validate_project_out(op, tiny)
    problems = _validate_indices(op, tiny.shapes[op['tensor']])
    if 'factor' in op:
        problems += _validate_multiplier('factor', op['factor'])
    return problems


def _check_fields(obj, required, optional, label, ignore=()):
    problems = [f"{label} is missing '{key}'" for key in required if key not in obj]
    allowed = set(required) | set(optional) | set(ignore)
    problems += [f"{label} has an unknown field '{key}'" for key in sorted(set(obj) - allowed)]
    return problems


def _validate_zero_head(op, tiny):
    layer, head = op['layer'], op['head']
    if not _is_int(layer) or not tiny.has_heads(layer):
        if len(tiny.head_layers()) == len(tiny.layer_types):
            return [f'layer {layer} does not exist: the layers are 0 to {len(tiny.layer_types) - 1}']
        return [f'layer {layer} is not a full-attention layer']
    if not _is_int(head) or not 0 <= head < tiny.num_attention_heads:
        return [f'head {head} is out of range: layer {layer} has heads 0 to {tiny.num_attention_heads - 1}']
    return []


def _validate_editable_tensor(name, tiny):
    if name in tiny.frozen:
        return [f"tensor '{name}' is frozen and cannot be edited"]
    if name not in tiny.names:
        return [f"tensor '{name}' is not an editable tensor"]
    return []


def _validate_indices(op, shape):
    """The rows or cols of an operation against the tensor's shape ('all' is project_out's only)."""
    key, axis, singular = ('cols', 1, 'col') if 'cols' in op else ('rows', 0, 'row')
    indices = op[key]
    if not isinstance(indices, list) or not indices:
        return [f"'{key}' must be a non-empty list of indices"]
    if len(indices) > cfg.MAX_INDICES:
        return [f"'{key}' has {len(indices)} indices, at most {cfg.MAX_INDICES} are allowed"]
    if axis >= len(shape):
        return [f"shape {shape} has no columns, so 'cols' cannot be used on this tensor"]
    return [f'{singular} {index} is out of range for shape {shape}'
            for index in indices if not _is_int(index) or not 0 <= index < shape[axis]]


def _validate_project_out(op, tiny):
    name = op['tensor']
    is_bias = tiny.is_residual_bias(name)
    if not tiny.is_residual_writer(name) and not is_bias:
        return [f"project_out cannot be used on tensor '{name}': {_residual_writers_sentence(tiny)}"]
    problems = _validate_multiplier('strength', op['strength'])
    if is_bias and op['cols'] != 'all':
        problems.append('\'cols\' must be "all" when project_out is used on a bias vector')
    elif op['cols'] != 'all':
        problems += _validate_indices({'cols': op['cols']}, tiny.shapes[name])
    return problems + _validate_direction(op['direction'], tiny)


def _validate_multiplier(key, value):
    if not _is_number(value) or not 0 <= value <= cfg.MAX_FACTOR:
        return [f"'{key}' must be a number between 0 and {cfg.MAX_FACTOR:g}, got {value!r}"]
    return []


def _optional_direction_fields(kind, tiny):
    optional = DIRECTION_FIELDS[kind][1]
    if kind in _UNEMBED_TYPES and tiny.entry['post_norms']:
        optional += ('fold_post_norm',)
    return optional


def _validate_direction(direction, tiny):
    if not isinstance(direction, dict):
        return ['the direction must be a JSON object']
    kind = direction.get('type')
    if kind not in DIRECTION_FIELDS:
        return [f"unknown direction type {kind!r}; the types are {', '.join(DIRECTION_FIELDS)}"]
    problems = _check_fields(direction, DIRECTION_FIELDS[kind][0], _optional_direction_fields(kind, tiny),
                             f"direction '{kind}'", ignore=('type',))
    if problems:
        return problems
    for key in ('fold_final_norm', 'fold_post_norm'):
        if key in direction and not isinstance(direction[key], bool):
            problems.append(f"'{key}' must be true or false")
    if kind == 'unembed_token':
        return problems + _validate_token(direction['token'], tiny)
    if kind == 'unembed_mean':
        return problems + _validate_token_list(direction['tokens'], tiny)
    return problems + _validate_svd(direction, tiny)


def _validate_token(token, tiny):
    if not isinstance(token, str):
        return [f'token {token!r} must be a string']
    n_ids = len(tiny.encode(token))
    if n_ids != 1:
        return [f'token {token!r} encodes to {n_ids} token ids, it must encode to exactly one']
    return []


def _validate_token_list(tokens, tiny):
    if not isinstance(tokens, list) or not tokens:
        return ["'tokens' must be a non-empty list of strings"]
    if len(tokens) > cfg.MAX_INDICES:
        return [f"'tokens' has {len(tokens)} entries, at most {cfg.MAX_INDICES} are allowed"]
    return [problem for token in tokens for problem in _validate_token(token, tiny)]


def _validate_svd(direction, tiny):
    name, component = direction['tensor'], direction['component']
    if not tiny.is_residual_writer(name):
        return [f"direction 'svd' cannot be used on tensor '{name}': {_residual_writers_sentence(tiny)}"]
    n_components = min(tiny.shapes[name])
    if not _is_int(component) or not 0 <= component < n_components:
        return [f'component {component} is out of range: the tensor has components 0 to {n_components - 1}']
    return []


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ------------------------------------------------------------------------------ application

def apply(spec, tiny, edited_dir):
    """Apply a validated spec in order, in float32, on copies, then save each touched tensor.

    Returns (record, errors). record is the {"kind": "spec"} line for tool_calls.jsonl. If the
    edit rules refuse any tensor (or a direction has no length), errors says why and nothing is
    kept in edited_dir.
    """
    working, directions = {}, []
    for i, op in enumerate(spec['edits']):
        name = _target_tensor(op, tiny)
        if name not in working:
            working[name] = tiny.load(name)
        if op['op'] == 'project_out':
            u, source = resolve_direction(op['direction'], tiny, target=name)
            if u is None:
                return None, [f'edits[{i}]: {source}']
            directions.append({'edit': i, **source})
            _project_out(working[name], u, op['strength'], op['cols'])
        else:
            _apply_simple_op(op, working[name], tiny)
    saved, unchanged, relative_change, errors = _save_all(working, tiny, edited_dir)
    if errors:
        return None, errors
    record = {'kind': 'spec', 'spec': spec, 'directions': directions, 'saved': saved,
              'no_edit': unchanged, 'relative_change': relative_change}
    return record, []


def resolve_direction(direction, tiny, target=None):
    """(unit vector, source record), or (None, a sentence saying why it has no direction).

    target: the tensor the direction is projected out of. It matters only for a model with
    post-block norms, where an unembed direction is by default folded through the norm that
    follows `target` (fold_post_norm).
    """
    if direction['type'] == 'svd':
        vector = _left_singular_vector(direction['tensor'], direction['component'], tiny)
        source = {'type': 'svd', 'tensor': direction['tensor'], 'component': direction['component']}
    else:
        tokens = [direction['token']] if direction['type'] == 'unembed_token' else direction['tokens']
        fold = direction.get('fold_final_norm', False)
        vector, token_ids = _unembed_mean(tokens, fold, tiny)
        source = {'type': direction['type'], 'tokens': tokens, 'token_ids': token_ids, 'fold_final_norm': fold}
        post_norm = tiny.post_norm_name(target) if target is not None else None
        if post_norm is not None:
            source['fold_post_norm'] = direction.get('fold_post_norm', True)
            if source['fold_post_norm']:
                # The norm writes diag(1 + g) y / rms(y) to the residual stream, so the part of the
                # tensor's output y that reaches direction u is its component along (1 + g) * u.
                vector = vector * (1.0 + tiny.load(post_norm))
                source['post_norm'] = post_norm
    length = float(vector.norm())
    if length == 0.0:
        return None, f"direction {direction['type']} has zero length"
    return vector / length, source


def _unembed_mean(tokens, fold_final_norm, tiny):
    # Input embedding and output layer are tied, so the output-embedding rows are the embedding's.
    token_ids = [tiny.encode(token)[0] for token in tokens]
    rows = tiny.load(tiny.entry['embedding'])[token_ids]
    if fold_final_norm:
        gain = tiny.load(tiny.entry['final_norm'])
        rows = rows * (1.0 + gain if tiny.entry['final_norm_kind'] == 'rms_one_plus' else gain)
    return rows.mean(dim=0), token_ids


def _left_singular_vector(name, component, tiny):
    # Always the original tensor, whatever earlier operations of this spec did to it.
    u, _, _ = torch.linalg.svd(tiny.load(name), full_matrices=False)
    return u[:, component]


def _target_tensor(op, tiny):
    return tiny.attention_output_name(op['layer']) if op['op'] == 'zero_head' else op['tensor']


def _apply_simple_op(op, w, tiny):
    kind = op['op']
    if kind == 'zero_head':
        first = op['head'] * tiny.head_dim
        w[:, first:first + tiny.head_dim] = 0.0
    elif kind == 'zero_columns':
        w[:, op['cols']] = 0.0
    elif kind == 'zero_rows':
        w[op['rows']] = 0.0
    elif kind == 'scale_columns':
        w[:, op['cols']] *= op['factor']
    elif kind == 'scale_rows':
        w[op['rows']] *= op['factor']


def _project_out(w, u, strength, cols):
    if w.dim() == 1:  # a bias vector: validated to come with cols "all"
        w -= strength * u * (u @ w)
        return
    selected = list(range(w.shape[1])) if cols == 'all' else cols
    block = w[:, selected]
    w[:, selected] = block - strength * torch.outer(u, u @ block)


def _save_all(working, tiny, edited_dir):
    """Save every working tensor. On a refusal, remove what was saved and report the refusal."""
    saved, unchanged, relative_change = [], [], {}
    for name, tensor in working.items():
        original = tiny.load(name)
        try:
            was_saved = edits.save_edit(edited_dir, name, tensor, original, tiny.names, tiny.frozen)
        except edits.EditRefused as exc:
            for already_saved in saved:
                edits.drop_edit(edited_dir, already_saved)
            return [], [], {}, [str(exc)]
        (saved if was_saved else unchanged).append(name)
        relative_change[name] = float((tensor - original).norm() / original.norm())
    return saved, unchanged, relative_change, []
