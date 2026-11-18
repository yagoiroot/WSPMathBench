The dominant failure is non-termination, not simply incorrect arithmetic: 226 of 256 answers hit the token limit, often after reaching the right calculation and then entering a “Wait, …” self-correction loop. I will therefore avoid global temperature-like changes and avoid weakening the model’s memory indiscriminately.

The program uses a static, contrastive logit-lens analysis. It reads—but never edits—the tied embedding and final norm to construct output directions for “Wait” variants and the token beginning/completing `\boxed`. Each direction is centered against common tokens from the practice solutions, making it token-specific rather than a common logit or sharpness direction. For every late MLP `down_proj` column and attention `out_proj` column, its coefficient on a “Wait” direction measures whether that feature writes or suppresses the loop token. The edit:

- removes positive “Wait”-writing components with strength increasing toward the final layer;
- gives additional ablation to the most token-specific MLP neurons and attention heads;
- slightly strengthens components that already actively suppress “Wait”;
- amplifies existing, specific `\boxed`-writing components in the last few layers.

These are rank-one, direction-specific projections: activation-producing `gate_proj` and `up_proj` weights remain unchanged, and each writer’s components orthogonal to the targeted tokens are preserved. No forward pass, activation collection, gradient, or iterative fitting is used. The program prints the selected neurons/heads, specificity scores, energy reductions, and edit sizes as evidence.

```python
import os
import gc
import json
import math
import random
import traceback
from collections import Counter

import torch
from safetensors import safe_open

# All computations below are closed-form weight-space operations.
# No network activations are computed and no parameters are fitted.
torch.set_num_threads(min(8, os.cpu_count() or 4))

MODEL_ROOT = "<MODELS_DIR>/Qwen3.5-0.8B"
PRACTICE_ROOT = (
    "<BENCHMARK_DIR>/practice"
)
VOCAB_SIZE = 248320
WIDTH = 1024


def read_model_tensor(name):
    """Read a tensor for analysis only; editable tensors still come from load()."""
    paths = sorted(
        os.path.join(MODEL_ROOT, f)
        for f in os.listdir(MODEL_ROOT)
        if f.endswith(".safetensors")
    )
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError("Could not locate " + name)


def encode(text):
    return [int(x) for x in tok.encode(text)]


def decode_one(idx):
    try:
        return tok.decode([int(idx)])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Obtain a math-text token basis from the references. This is used only to
# center token directions and estimate specificity; no solutions are fitted.
# ---------------------------------------------------------------------------
counts = Counter()
for filename in ("gsm8k_train.jsonl", "math_train.jsonl"):
    path = os.path.join(PRACTICE_ROOT, filename)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no >= 64:
                    break
                try:
                    obj = json.loads(line)
                    text = (
                        obj.get("problem", "")
                        + "\n"
                        + obj.get("solution", "")
                        + "\n"
                        + str(obj.get("answer", ""))
                    )[:10000]
                    counts.update(encode(text))
                except Exception:
                    continue
    except Exception as exc:
        print("Practice token-basis fallback:", filename, repr(exc))

common_ids = [
    int(i) for i, _ in counts.most_common(512)
    if 0 <= int(i) < VOCAB_SIZE
]
rng = random.Random(1729)
random_ids = rng.sample(range(VOCAB_SIZE), 512)


# Accept only pieces that actually contain the relevant substring. In
# particular, do not boost a bare backslash if \boxed is split into pieces.
wait_ids = []
for spelling in ("Wait", " wait", "wait", "WAIT"):
    try:
        ids = encode(spelling)
        for idx in ids:
            piece = decode_one(idx)
            if "wait" in piece.lower() and idx not in wait_ids:
                wait_ids.append(idx)
        if len(ids) == 1 and ids[0] not in wait_ids:
            wait_ids.append(ids[0])
    except Exception:
        pass

box_id = None
box_source = None
for spelling in ("\\boxed", "boxed", "\\boxed{"):
    try:
        ids = encode(spelling)
        if len(ids) == 1 and "box" in decode_one(ids[0]).lower():
            box_id = ids[0]
            box_source = spelling
            break
    except Exception:
        pass

if box_id is None:
    try:
        ids = encode("\\boxed")
        for idx in ids:
            if "box" in decode_one(idx).lower():
                box_id = idx
                box_source = "\\boxed"
                break
    except Exception:
        pass

print("Wait token pieces:",
      [(int(i), repr(decode_one(i))) for i in wait_ids])
print("Box token piece:",
      None if box_id is None else (int(box_id), repr(decode_one(box_id))),
      "source:", repr(box_source))


# ---------------------------------------------------------------------------
# Contrastive output directions.
#
# If E_t is a tied-unembedding row and gamma is the final RMSNorm scale, then
# p_t = gamma * E_t is the direct logit-lens direction. Subtracting the mean
# math-token direction removes the large common logit component. Consequently,
# editing along p_t-mean changes token-relative writing, not global sharpness.
# ---------------------------------------------------------------------------
embedding = read_model_tensor(
    "model.language_model.embed_tokens.weight"
).to(torch.float32)
final_gamma = read_model_tensor(
    "model.language_model.norm.weight"
).to(torch.float32).reshape(-1)

assert embedding.shape[1] == WIDTH
assert final_gamma.numel() == WIDTH

common_ids = [i for i in common_ids if i < embedding.shape[0]]
wait_ids = [i for i in wait_ids if i < embedding.shape[0]]
if box_id is not None and box_id >= embedding.shape[0]:
    box_id = None

if not common_ids:
    common_ids = random_ids[:256]

center_ids = common_ids[:384]
center_basis = embedding[center_ids] * final_gamma
mu = center_basis.mean(dim=0)

basis_ids = list(dict.fromkeys(common_ids[:384] + random_ids[:384]))
basis = embedding[basis_ids] * final_gamma - mu
basis_norm = basis.norm(dim=1)
valid_basis = (
    torch.isfinite(basis).all(dim=1)
    & (basis_norm > 1e-8)
)
basis = basis[valid_basis]
basis_norm = basis_norm[valid_basis]
if basis.shape[0] == 0:
    raise RuntimeError("Empty token-comparison basis")
basis = basis[:768]
basis_norm = basis_norm[:768]
unit_basis = basis / basis_norm[:, None].clamp_min(1e-12)

raw_wait = []
for idx in wait_ids:
    direction = embedding[idx] * final_gamma - mu
    norm = direction.norm()
    if torch.isfinite(norm) and norm > 1e-8:
        raw_wait.append(direction / norm)

if not raw_wait:
    raise RuntimeError("No usable Wait output direction")

# Combine spelling variants without applying the same highly correlated edit
# several times. A second component is retained only if it carries substantial
# independent spelling information.
raw_wait_matrix = torch.stack(raw_wait)
_, singular_values, right_vectors = torch.linalg.svd(
    raw_wait_matrix, full_matrices=False
)
wait_directions = []
for j in range(min(2, right_vectors.shape[0])):
    if j > 0 and singular_values[j] < 0.38 * singular_values[0]:
        continue
    direction = right_vectors[j].clone()
    if torch.dot(direction, raw_wait_matrix.sum(dim=0)) < 0:
        direction = -direction
    wait_directions.append(direction / direction.norm().clamp_min(1e-12))

box_direction = None
if box_id is not None:
    candidate = embedding[box_id] * final_gamma - mu
    # Do not let subsequent box amplification reintroduce a Wait writer.
    for direction in wait_directions:
        candidate = candidate - torch.dot(candidate, direction) * direction
    norm = candidate.norm()
    if torch.isfinite(norm) and norm > 1e-8:
        box_direction = candidate / norm

print("Wait singular values:", [float(x) for x in singular_values])
print("Number of orthogonal Wait directions:", len(wait_directions))
print("Box direction available:", box_direction is not None)

del embedding, center_basis, basis, raw_wait_matrix, right_vectors
gc.collect()


# ---------------------------------------------------------------------------
# Token-specific writer ranking. Columns are individual SwiGLU neurons for
# down_proj, and individual head/value features for attention out projections.
# The z-score compares a column's token cosine with the same column's cosines
# over the practice/random token basis.
# ---------------------------------------------------------------------------
def specific_columns(matrix, direction, k):
    ncols = matrix.shape[1]
    norms = matrix.norm(dim=0)
    cosine = (direction @ matrix) / norms.clamp_min(1e-12)
    cosine = torch.where(
        norms > 1e-10,
        cosine,
        torch.full_like(cosine, -float("inf")),
    )

    pool_size = min(ncols, max(96, 4 * k))
    pool_values, pool = torch.topk(cosine, pool_size)
    keep = torch.isfinite(pool_values)
    pool_values = pool_values[keep]
    pool = pool[keep]
    if pool.numel() == 0:
        return torch.empty(0, dtype=torch.long), []

    normalized_columns = matrix[:, pool] / norms[pool].clamp_min(1e-12)
    comparison = unit_basis @ normalized_columns
    mean = comparison.mean(dim=0)
    std = comparison.std(dim=0, unbiased=False).clamp_min(1e-5)
    zscore = (pool_values - mean) / std

    # Both high relative specificity and high absolute cosine are desirable.
    quality = zscore + 4.0 * pool_values
    order = torch.argsort(quality, descending=True)
    take = order[:min(k, order.numel())]
    selected = pool[take]
    evidence = [
        (
            int(pool[pos]),
            float(pool_values[pos]),
            float(zscore[pos]),
        )
        for pos in take[:10]
    ]
    return selected, evidence


def mask_from_indices(ncols, indices):
    mask = torch.zeros(ncols, dtype=torch.float32)
    if indices is not None and len(indices) > 0:
        mask[indices] = 1.0
    return mask


def positive_wait_energy(matrix):
    energy = 0.0
    for direction in wait_directions:
        coeff = direction @ matrix
        energy += float(coeff.clamp_min(0).square().sum())
    return energy


MLP_STRENGTH = {
    18: 0.18, 19: 0.22, 20: 0.32,
    21: 0.46, 22: 0.68, 23: 0.92,
}
ATTN_STRENGTH = {
    18: 0.12, 19: 0.16, 20: 0.24,
    21: 0.36, 22: 0.58, 23: 0.86,
}
BOX_BASE = {20: 0.02, 21: 0.04, 22: 0.07, 23: 0.10}
BOX_TOP = {20: 0.10, 21: 0.18, 22: 0.38, 23: 0.70}


def edit_writer_matrix(name, layer, kind, head_size=None):
    matrix = load(name).to(torch.float32).clone()
    original = matrix.clone()
    ncols = matrix.shape[1]
    is_mlp = kind == "mlp"
    strength = (MLP_STRENGTH if is_mlp else ATTN_STRENGTH)[layer]

    initial_energy = positive_wait_energy(original)
    top_wait, wait_evidence = specific_columns(
        original, wait_directions[0], 48 if is_mlp else 24
    )
    top_wait_mask = mask_from_indices(ncols, top_wait)
    implicated_heads = set()

    for direction_index, direction in enumerate(wait_directions):
        # The second independent spelling direction receives a slightly milder
        # edit. Directions are orthonormal, so later edits do not undo earlier
        # ones.
        local_strength = strength * (1.0 if direction_index == 0 else 0.80)
        coeff = direction @ matrix
        positive = coeff.clamp_min(0)
        negative = coeff.clamp_max(0)

        # Remove token-writing components, leaving every orthogonal component
        # of every column intact.
        delta = -local_strength * torch.outer(direction, positive)

        # Strengthen already-existing brakes modestly. This can actively oppose
        # a loop direction arriving through the residual stream, rather than
        # merely refusing to add more of it.
        brake_rate = (0.20 if is_mlp else 0.12) * local_strength
        delta = delta + brake_rate * torch.outer(direction, negative)

        extra_rate = (0.06 if is_mlp else 0.09) * local_strength
        if is_mlp:
            if direction_index == 0:
                delta = delta - extra_rate * torch.outer(
                    direction, positive * top_wait_mask
                )
        else:
            # Additional ablation is restricted to the heads with the greatest
            # positive Wait-writing energy, not spread uniformly over heads.
            heads = ncols // head_size
            energy_by_head = positive.view(heads, head_size).square().sum(dim=1)
            head_count = 2 if heads == 8 else 4
            top_heads = torch.topk(
                energy_by_head, min(head_count, heads)
            ).indices
            head_mask = torch.zeros(ncols, dtype=torch.float32)
            for head in top_heads:
                h = int(head)
                implicated_heads.add(h)
                head_mask[h * head_size:(h + 1) * head_size] = 1.0
            delta = delta - extra_rate * torch.outer(
                direction, positive * head_mask
            )

        matrix = matrix + delta

    box_evidence = []
    if box_direction is not None and layer >= 20:
        top_box, box_evidence = specific_columns(
            matrix, box_direction, 48 if is_mlp else 24
        )
        top_box_mask = mask_from_indices(ncols, top_box)
        coeff = box_direction @ matrix
        positive = coeff.clamp_min(0)

        # Attention receives a smaller finalization edit than the last MLPs.
        kind_factor = 1.0 if is_mlp else 0.50
        base_boost = BOX_BASE[layer] * kind_factor
        top_boost = BOX_TOP[layer] * kind_factor

        # A mild increase for all existing box writers, plus a substantial
        # increase only for the most box-specific neurons/head features.
        matrix = matrix + base_boost * torch.outer(box_direction, positive)
        matrix = matrix + top_boost * torch.outer(
            box_direction, positive * top_box_mask
        )

    # Safety limit on total Frobenius perturbation. Normally the edit is far
    # below this because it occupies only a few directions in a 1024-dimensional
    # output space.
    change = matrix - original
    original_norm = original.norm().clamp_min(1e-12)
    relative_change = float(change.norm() / original_norm)
    if relative_change > 0.22:
        matrix = original + change * (0.22 / relative_change)
        relative_change = 0.22

    if not torch.isfinite(matrix).all():
        raise RuntimeError("Non-finite edit for " + name)

    final_energy = positive_wait_energy(matrix)
    reduction = (
        100.0 * (1.0 - final_energy / initial_energy)
        if initial_energy > 1e-20 else 0.0
    )

    print("\nEDIT", name)
    print("  relative Frobenius change:", relative_change)
    print("  positive Wait-writing energy reduction (%):", reduction)
    print("  specific Wait columns [(index, cosine, z), ...]:", wait_evidence)
    if implicated_heads:
        print("  high Wait-writing heads:", sorted(implicated_heads))
    if box_evidence:
        print("  specific box columns [(index, cosine, z), ...]:", box_evidence)

    save_edit(name, matrix)


# Only the final six layers are edited. Earlier reasoning and memory circuits
# are untouched, while the final two layers receive the decisive intervention.
for layer in (18, 19, 20, 21, 22, 23):
    prefix = f"model.language_model.layers.{layer}"

    jobs = [
        (f"{prefix}.mlp.down_proj.weight", "mlp", None),
    ]
    if layer in (19, 23):
        jobs.append((f"{prefix}.self_attn.o_proj.weight", "attention", 256))
    else:
        jobs.append((f"{prefix}.linear_attn.out_proj.weight", "attention", 128))

    for name, kind, head_size in jobs:
        try:
            edit_writer_matrix(name, layer, kind, head_size)
        except Exception as exc:
            # Preserve all successful edits if one tensor proves problematic.
            print("FAILED", name, repr(exc))
            traceback.print_exc()

print("\nSaved editable tensors:")
for name in edited():
    print(" ", name)
print("Total saved tensors:", len(edited()))
```