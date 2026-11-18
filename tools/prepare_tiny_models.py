"""Copy a tiny model out of the Hugging Face cache into a plain folder, converting it where needed.

Where it sits in the flow: run once per tiny model, by hand, before anything else (the one-time
setup, docs/SETUP.md), after the model has been downloaded into the Hugging Face cache at the
revision the registry names. In the Python environment of requirements.txt, from the repository
root:

    python tools/prepare_tiny_models.py [KEY ...] [--force]

With no KEY it prepares every registry model that names a `source_snapshot`. An existing target
folder is left alone unless --force is given. Qwen3.5-0.8B names none: its files are downloaded
straight into its model folder as plain files.

Why a plain folder: the cache holds symlinks into blobs/, and under WSL the kernel and vLLM read
the model through /mnt/<drive>/, where plain files are the safe form.

Two models are converted, because every tiny model runs in bfloat16 and the benchmark's base must
be the numbers the running model actually uses:
    MobileLLM-R1-950M   float32 safetensors -> bfloat16 safetensors
    TinyStories-33M     float32 pytorch_model.bin (a pickle) -> bfloat16 safetensors, without the
                        two non-weight buffers per layer (attn.attention.bias, a boolean mask, and
                        masked_bias). The pickle is read with weights_only=True, which executes
                        nothing, and is not copied into the folder.
For a converted model, config.json's dtype field is set to bfloat16 and wsp_source.json records
the source revision, the source file's SHA-256 and what was done.

Reads the cache snapshot. Writes <model_dir>/ (cfg.TINY_MODEL_REGISTRY[key]['model_dir']).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from wspmathbench import config as cfg  # noqa: E402

SOURCE_RECORD_FILE = 'wsp_source.json'
WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')
NOT_COPIED = ('.gitattributes',)
# Buffers of GPT-Neo's attention that are not weights: a causal mask and a fill value.
NON_WEIGHT_SUFFIXES = ('.attn.attention.bias', '.attn.attention.masked_bias')
DTYPE_KEYS = ('torch_dtype', 'dtype')


def prepare(key, force=False):
    entry = cfg.TINY_MODEL_REGISTRY[key]
    source, target = entry.get('source_snapshot'), entry['model_dir']
    if source is None:
        print(f'{key}: no source_snapshot in the registry, nothing to prepare')
        return
    if not os.path.isdir(source):
        raise FileNotFoundError(f"{key}: the cache snapshot {source} does not exist. Download "
                                f"{entry['hf_repo']} at revision {entry['revision']} first.")
    if os.path.exists(target):
        if not force:
            print(f'{key}: {target} already exists, left alone (use --force to rebuild it)')
            return
        shutil.rmtree(target)
    os.makedirs(target)

    for file_name in sorted(os.listdir(source)):
        path = os.path.join(source, file_name)
        if os.path.isfile(path) and file_name not in WEIGHT_FILES + NOT_COPIED:
            shutil.copyfile(path, os.path.join(target, file_name))  # follows the cache's symlinks

    source_weights = next(os.path.join(source, f) for f in WEIGHT_FILES if os.path.exists(os.path.join(source, f)))
    target_weights = os.path.join(target, cfg.CHECKPOINT_WEIGHTS_FILE)
    conversion = entry.get('conversion')
    if conversion is None:
        shutil.copyfile(source_weights, target_weights)
    else:
        record = _convert_to_bf16(source_weights, target_weights)
        _set_config_dtype(os.path.join(target, cfg.MODEL_CONFIG_FILE))
        record = {'hf_repo': entry['hf_repo'], 'revision': entry['revision'],
                  'source_file': os.path.basename(source_weights), 'conversion': conversion, **record}
        with open(os.path.join(target, SOURCE_RECORD_FILE), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=1)
        print(f'{key}: {json.dumps(record)}')
    size_gb = os.path.getsize(target_weights) / 2**30
    print(f'{key}: {target} ready, weights {size_gb:.2f} GB')


def _convert_to_bf16(source_weights, target_weights):
    if source_weights.endswith('.bin'):
        state = torch.load(source_weights, map_location='cpu', weights_only=True)
    else:
        state = load_file(source_weights)
    dropped = sorted(name for name in state if name.endswith(NON_WEIGHT_SUFFIXES))
    converted = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in state.items()
                 if name not in dropped and tensor.is_floating_point()}
    not_float = sorted(name for name in state if name not in dropped and name not in converted)
    if not_float:
        raise ValueError(f'Tensors that are neither weights nor known buffers: {not_float}')
    save_file(converted, target_weights, metadata={'format': 'pt'})
    return {'source_sha256': _sha256_file(source_weights), 'converted_sha256': _sha256_file(target_weights),
            'source_dtypes': sorted({str(t.dtype).replace('torch.', '') for t in state.values()}),
            'n_tensors': len(converted), 'dropped_buffers': dropped}


def _set_config_dtype(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    present = [key for key in DTYPE_KEYS if key in config]
    for key in present or DTYPE_KEYS[:1]:
        config[key] = 'bfloat16'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('keys', nargs='*', help='registry keys; default: every model with a source_snapshot')
    parser.add_argument('--force', action='store_true', help='rebuild a folder that already exists')
    args = parser.parse_args()
    for key in args.keys or [k for k, e in cfg.TINY_MODEL_REGISTRY.items() if e.get('source_snapshot')]:
        prepare(key, args.force)


if __name__ == '__main__':
    main()
