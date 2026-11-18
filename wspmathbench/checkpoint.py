"""Build a full edited checkpoint directory from the base model plus a session's edited/ folder.

Where it sits in the flow: step 7 of the session. Adapted from build_ckpt.py of Takehi et al.

Reads the base model directory and <session>/edited/*.npy. Writes <out_dir>/ (up to about 2 GB,
under cfg.CHECKPOINT_DIR): the base model's files, model.safetensors and edits.json. The result
is meant to load with `vllm serve <out_dir>` exactly like the base.

check_in_vllm, run by hand after a session, checks that it does: it starts
tools/check_checkpoint_vllm.py on the Linux side, which writes nothing.
"""
import json
import os
import shlex
import shutil
import subprocess

import torch
from safetensors.torch import load_file, save_file

from wspmathbench import config as cfg
from wspmathbench import edits


def build(tiny, edited_dir, out_dir, manifest, session_id=None, task=None):
    """Write the edited checkpoint to out_dir, which must not exist yet.

    manifest: edits.manifest of edited_dir; it is stored in edits.json with the tiny model,
    its base revision, the task and the session ID, which is how the checkpoint check and the
    scoring code know what the folder is. Returns out_dir.
    """
    if os.path.exists(out_dir):
        raise FileExistsError(f'{out_dir} already exists; a checkpoint is never built over another one.')
    os.makedirs(out_dir)
    _copy_base_files(tiny.model_dir, out_dir)

    state_dict = load_file(tiny.weights_path)
    for name in edits.edited(edited_dir):
        state_dict[name] = _cast_like(edits.load_edit(edited_dir, name), state_dict[name], name)
    save_file(state_dict, os.path.join(out_dir, cfg.CHECKPOINT_WEIGHTS_FILE), metadata={'format': 'pt'})

    record = {**manifest, 'tiny_model': tiny.key, 'task': task, 'base_revision': tiny.revision,
              'session_id': session_id}
    with open(os.path.join(out_dir, cfg.CHECKPOINT_RECORD_FILE), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    return out_dir


def check_in_vllm(checkpoint, tiny_model=None, n_per_source=2, max_tokens=4096, seed=0):
    """Load a checkpoint in its engine (vLLM, or transformers for TinyStories-33M) and answer a few
    practice problems, to check that the edited weights still load and still produce text. Not a
    score.

    checkpoint: a session ID (its folder under cfg.CHECKPOINT_DIR), or any model folder, for
        example the base model's for comparison. The same arguments give the same problems and
        sampling seeds.
    tiny_model: only for a folder without edits.json (an unedited model's): which registry model
        it is. None means Qwen3.5-0.8B. An edited checkpoint names its own tiny model.
    n_per_source: practice problems from each source of the task (2 GSM8K and 2 MATH by default).
    max_tokens: per answer, never more than the protocol's own limit. Qwen's 32,768 also works
        but takes much longer.

    Runs tools/check_checkpoint_vllm.py on the Linux side, on the GPU, so nothing else may be
    using the GPU.
    Takes a few minutes, mostly the engine starting up. Prints as it goes and returns True if the
    tool finished without an error.
    """
    model_dir = cfg.checkpoint_dir(checkpoint)
    if not os.path.isdir(model_dir) or not any(name.endswith('.safetensors') for name in os.listdir(model_dir)):
        raise FileNotFoundError(f'{model_dir} is not a model folder: it has no .safetensors file.')
    tool, model_wsl, practice_wsl = (shlex.quote(cfg.to_wsl_path(path)) for path in
                                     (os.path.join(cfg.TOOLS_DIR, 'check_checkpoint_vllm.py'),
                                      model_dir, cfg.PRACTICE_DIR))
    # The python path is left unquoted so bash expands its ~. -u: print as it goes, not at the end.
    command = (f'{cfg.VLLM_PYTHON} -u {tool} --model-dir {model_wsl} --practice-dir {practice_wsl} '
               f'--n-per-source {int(n_per_source)} --max-tokens {int(max_tokens)} --seed {int(seed)}')
    if tiny_model is not None:
        cfg.tiny_entry(tiny_model)  # raises for a name that is not in the registry
        command += f' --tiny-model {shlex.quote(tiny_model)}'
    return subprocess.run(cfg.shell_command(command)).returncode == 0


def _copy_base_files(model_dir, out_dir):
    # Top-level files only: the download's .cache folder is not part of the model. Weights files
    # are left behind: the checkpoint gets its own model.safetensors.
    for file_name in sorted(os.listdir(model_dir)):
        source = os.path.join(model_dir, file_name)
        if (os.path.isfile(source) and not file_name.endswith(cfg.WEIGHTS_EXTENSIONS)
                and file_name not in cfg.CHECKPOINT_NOT_COPIED):
            shutil.copyfile(source, os.path.join(out_dir, file_name))


def _cast_like(array, original, name):
    new = torch.from_numpy(array).to(original.dtype)
    if new.shape != original.shape:
        raise ValueError(f"Saved edit of '{name}' has shape {list(new.shape)}, the base has {list(original.shape)}.")
    return new.contiguous()
