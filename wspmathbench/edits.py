"""The edit rules, saving and dropping edits, and the edit manifest.

The rules (save_edit enforces them, for both tiers): only an editable tensor of the language
model may be edited, never a frozen one; the shape must not change; every value must be finite; and
an edit that only multiplies a whole tensor by a constant is refused. An edit equal to the
original is dropped, not saved.

Where it sits in the flow: both tiers save through `save_edit` (the oneshot_spec tier via
dsl.apply in step 4, the oneshot_code tier from inside the kernel), and step 5 of the
session calls `manifest`.

Reads and writes <session>/edited/<name>.npy (float32, one file per tensor).

No imports from the package: kernel_runner.py loads this file by its path, in a sandbox venv
that has only numpy, torch, safetensors and tokenizers.
"""
import hashlib
import os

import numpy as np
import torch

RESCALE_TOLERANCE = 1e-6  # of the pure-rescale rule; kept here because this file imports nothing from the package


class EditRefused(Exception):
    """An edit broke a rule. The message is one sentence that says which."""


def save_edit(edited_dir, name, tensor, original, names, frozen):
    """Check `tensor` against the rules and store it as edited_dir/<name>.npy.

    original: the tensor's original values as float32. It may be None only when `name` is not an
    editable tensor, because that name is refused before the values are looked at.
    Returns True if saved. Returns False if `tensor` equals the original: that is "no edit", so
    any earlier edit of this tensor is dropped and nothing is raised (rule 6).
    Raises EditRefused otherwise.
    """
    if name in frozen:
        raise EditRefused(f"tensor '{name}' is frozen and cannot be edited")
    if name not in names:
        raise EditRefused(f"tensor '{name}' is not an editable tensor")
    new = torch.as_tensor(tensor).detach().to(torch.float32)
    if list(new.shape) != list(original.shape):
        raise EditRefused(f"tensor '{name}' has shape {list(new.shape)}, the original has shape {list(original.shape)}")
    if not torch.isfinite(new).all():
        raise EditRefused(f"tensor '{name}' contains a value that is not finite")
    if torch.equal(new, original):
        drop_edit(edited_dir, name)
        return False
    if _is_pure_rescale(new, original):
        raise EditRefused(f"the edit of tensor '{name}' only multiplies the whole tensor by a constant")
    _write_npy(edited_dir, name, new.cpu().numpy())
    return True


def drop_edit(edited_dir, name):
    """Undo a save_edit. Does nothing if the tensor has no saved edit."""
    path = _edit_path(edited_dir, name)
    if os.path.exists(path):
        os.remove(path)


def edited(edited_dir):
    """The names with a saved edit, sorted."""
    if not os.path.isdir(edited_dir):
        return []
    return sorted(f[:-len('.npy')] for f in os.listdir(edited_dir) if f.endswith('.npy'))


def load_edit(edited_dir, name):
    """A saved edit as a float32 numpy array."""
    return np.load(_edit_path(edited_dir, name))


def manifest(edited_dir, tiny):
    """What the saved edits change, measured after casting back to each tensor's stored dtype.

    Returns {'tensors': [{'name', 'shape', 'dtype', 'entries_changed', 'relative_change',
    'sha256_original', 'sha256_edited'}, ...], 'n_tensors', 'entries_changed'}. `tiny` needs only
    a load_raw(name) method.
    """
    tensors = [_manifest_entry(edited_dir, name, tiny.load_raw(name)) for name in edited(edited_dir)]
    return {'tensors': tensors, 'n_tensors': len(tensors),
            'entries_changed': sum(t['entries_changed'] for t in tensors)}


def _manifest_entry(edited_dir, name, original):
    # Most tensors are bfloat16, but a few (A_log, linear_attn.norm) are stored as float32, so the
    # cast target is the tensor's own dtype rather than a fixed bfloat16.
    new = torch.from_numpy(load_edit(edited_dir, name)).to(original.dtype)
    original_float = original.to(torch.float64)
    return {
        'name': name,
        'shape': list(original.shape),
        'dtype': str(original.dtype).replace('torch.', ''),
        'entries_changed': int((new != original).sum()),
        'relative_change': float((new.to(torch.float64) - original_float).norm() / original_float.norm()),
        'sha256_original': _sha256(original),
        'sha256_edited': _sha256(new),
    }


def _is_pure_rescale(new, original):
    # The pure-rescale rule, in float64 so float32 rounding cannot hide or fake a rescale. An
    # all-zero `new` makes the ratio 0/0; it is a multiplication by 0, so it counts as a rescale.
    new, original = new.to(torch.float64), original.to(torch.float64)
    original_norm_squared = float((original * original).sum())
    if original_norm_squared == 0.0:
        return False
    c = float((new * original).sum()) / original_norm_squared
    new_norm = float(new.norm())
    if new_norm == 0.0:
        return True
    return c != 1.0 and float((new - c * original).norm()) / new_norm < RESCALE_TOLERANCE


def _write_npy(edited_dir, name, array):
    # Write then rename: a program killed at its time limit must never leave half a file behind,
    # because "edits saved before it are kept".
    os.makedirs(edited_dir, exist_ok=True)
    path = _edit_path(edited_dir, name)
    temporary = path + '.tmp'
    with open(temporary, 'wb') as f:
        np.save(f, array)
    os.replace(temporary, path)


def _edit_path(edited_dir, name):
    return os.path.join(edited_dir, name + '.npy')


def _sha256(tensor):
    # numpy has no bfloat16, so hash its bits as uint16. The bytes are the same as the stored ones.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
