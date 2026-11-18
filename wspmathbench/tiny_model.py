"""Reading the tiny model: tensor names and shapes, layer kinds, heads, the tokenizer, load(name).

Where it sits in the flow: step 1 of the session. Everything downstream (the prompt,
the validator, the edit rules, the checkpoint builder) gets its facts about the model from the
TinyModel returned by `open`, so no tensor name is typed anywhere else. What differs between
architectures (where the config sits, how layers are named, which layers have heads, how the
norms work) comes from the model's entry in cfg.TINY_MODEL_REGISTRY.

Reads the model directory (the safetensors header and tensors, config.json, tokenizer.json).
Writes nothing. Not used inside the oneshot_code kernel, which has its own small prelude.
"""
import io
import json
import os
import re
import struct
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from wspmathbench import config as cfg

_FLOAT_DTYPES = ('F16', 'BF16', 'F32', 'F64')


@dataclass
class TinyModel:
    """Facts about one tiny model. Built by `open`; do not construct it directly.

    names            editable tensors, in layer order (language-model tensors minus frozen)
    language_names   names plus the frozen language-model tensors (readable, not editable)
    shapes           name -> list of ints, for language_names
    frozen           the registry's frozen names, including any absent from this file
    layer_types      each layer's kind, as the model's config names it, indexed by layer
    """
    key: str
    entry: dict
    model_dir: str
    weights_path: str
    config: dict
    names: list
    language_names: list
    shapes: dict
    frozen: list
    layer_types: list
    _tokenizer: object = field(default=None, repr=False)

    @property
    def revision(self):
        return self.entry['revision']

    @property
    def width(self):
        """The width of the residual stream: the row count of a tensor that writes to it."""
        return self.shapes[self.residual_writers()[0]][0]

    @property
    def num_attention_heads(self):
        return self.config[self.entry['heads_key']]

    @property
    def head_dim(self):
        """The number of attention-output columns that belong to one head."""
        return self.shapes[self.attention_output_name(self.head_layers()[0])][1] // self.num_attention_heads

    def has_heads(self, layer):
        """Whether zero_head applies to this layer (its attention output is split into heads)."""
        return 0 <= layer < len(self.layer_types) and self.layer_types[layer] in self.entry['head_kinds']

    def head_layers(self):
        return [layer for layer in range(len(self.layer_types)) if self.has_heads(layer)]

    def layer_of(self, name):
        """The layer index in a tensor name, or None for a tensor outside the layers."""
        return _layer_of(name, self.entry)

    def layer_label(self, layer):
        """How the prompt's tensor table names this layer's kind."""
        return self.entry['layer_kind_labels'][self.layer_types[layer]]

    def is_residual_writer(self, name):
        """A weight matrix whose rows are the residual stream's dimensions."""
        return name in self.names and name.endswith(tuple(self.entry['residual_writer_suffixes']))

    def is_residual_bias(self, name):
        """A bias vector added to the residual stream with a residual writer's output."""
        suffixes = tuple(self.entry['residual_bias_suffixes'])
        return bool(suffixes) and name in self.names and name.endswith(suffixes)

    def residual_writers(self):
        return [name for name in self.names if self.is_residual_writer(name)]

    def post_norm_name(self, name):
        """The norm between residual writer `name` and the residual stream, or None."""
        for writer_suffix, norm_suffix in self.entry['post_norms'].items():
            if name.endswith(writer_suffix):
                return name[:-len(writer_suffix)] + norm_suffix
        return None

    def attention_output_name(self, layer):
        """The tensor that zero_head edits. Its columns are the attention heads' outputs."""
        return self.entry['layer_stem'].format(layer=layer) + self.entry['attention_output_suffix']

    def load_raw(self, name):
        """A tensor as stored in the file (a copy, in its own dtype, usually bfloat16)."""
        with safe_open(self.weights_path, framework='pt') as f:
            return f.get_tensor(name).clone()

    def load(self, name):
        """A tensor as float32 (a copy with the original values)."""
        return self.load_raw(name).to(torch.float32)

    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, cfg.MODEL_TOKENIZER_FILE))
        return self._tokenizer

    def encode(self, text):
        """Token ids of `text`, without special tokens."""
        return self.tokenizer().encode(text, add_special_tokens=False).ids


def open(key):  # noqa: A001 - read as tiny_model.open; this module uses io.open for files
    """The TinyModel for a key of cfg.TINY_MODEL_REGISTRY. Raises early if its files are missing."""
    entry = cfg.tiny_entry(key)
    model_dir = entry['model_dir']
    if not os.path.isdir(model_dir):
        how = ('Build it with tools/prepare_tiny_models.py.' if entry.get('source_snapshot') else
               f"Download {entry['hf_repo']} at revision {entry['revision']} there.")
        raise FileNotFoundError(f"The tiny model's folder {model_dir} does not exist. {how}")
    weights_path = _find_weights_file(model_dir)
    header = _read_safetensors_header(weights_path)
    with io.open(os.path.join(model_dir, cfg.MODEL_CONFIG_FILE), encoding='utf-8') as f:
        config = json.load(f)
    if entry['config_root'] is not None:
        config = config[entry['config_root']]

    frozen = list(entry['frozen'])
    # Weights only: a file may also hold buffers that are not floating point (masks, counters).
    language_names = sorted((n for n in header if n.startswith(entry['editable_prefix'])
                             and header[n]['dtype'] in _FLOAT_DTYPES), key=lambda n: _layer_order(n, entry))
    return TinyModel(
        key=key, entry=entry, model_dir=model_dir, weights_path=weights_path, config=config,
        names=[n for n in language_names if n not in frozen],
        language_names=language_names,
        shapes={n: list(header[n]['shape']) for n in language_names},
        frozen=frozen,
        layer_types=list(config[entry['layer_kinds_key']]))


def _layer_of(name, entry):
    stem, _, _ = entry['layer_stem'].partition('{layer}')
    match = re.match(re.escape(stem) + r'(\d+)\.', name)
    return int(match.group(1)) if match else None


def _layer_order(name, entry):
    # Numeric layer order (the file's own order is arbitrary); tensors outside the layers go last.
    layer = _layer_of(name, entry)
    return (layer is None, layer if layer is not None else 0, name)


def _find_weights_file(model_dir):
    # Qwen's release ships one file named like a shard (model.safetensors-00001-of-00001.safetensors).
    found = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one .safetensors file in {model_dir}, found {found}.')
    return os.path.join(model_dir, found[0])


def _read_safetensors_header(path):
    with io.open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop('__metadata__', None)
    return header
