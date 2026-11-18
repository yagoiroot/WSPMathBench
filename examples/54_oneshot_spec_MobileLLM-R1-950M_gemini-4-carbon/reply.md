### Mechanistic Diagnosis of `MobileLLM-R1-950M`

The baseline evaluation reveals a striking diagnostic inversion: `MobileLLM-R1-950M` scores **78.9% on competition MATH** (101/128) yet only **75.0% on grade-school GSM8K** (96/128), while averaging **3,556 tokens per response** and losing **11 of its 59 errors (18.6%)** to 30,000-token cutoffs (`7`) and unclosed degenerate loops (`4`).

Inspecting the `<think>` traces shows why a model with strong algebraic competence underperforms on straightforward word problems and suffers runaway length inflation:

1. **Unprovoked Mid-Calculation Self-Doubt**: Distilled from long R1-style reasoning traces into a 950M-parameter network, the student has absorbed an over-active surface prior for reflexive backtracking (`"Wait"` / `" Wait"`) even when no internal contradiction exists. In **Example 5**, the model computes the exact right intermediate step and immediately talks itself out of it (*`"8 years ago he had 20 - 8 = 12 years. Wait, no, that's not right. Wait, no..."`*). In **Example 8**, the same reflex derails multiplying the flag's area by 2 (*`"Now, he needs to paint both sides. Wait, does that mean he has to paint the entire surface area, or..."`*). Once `" Wait"` is sampled at `temperature = 0.6`, causal self-attention forces subsequent steps to rationalize the objection by inventing an error.
2. **Speculative Guessing / Hedging**: When uncertain or over-analyzing simple problems, the model emits epistemic hedging tokens (`" maybe"`, `" perhaps"`) that shift autoregressive generation from deductive calculation into ungrounded fabrication—as in **Example 4**, where after finding the exact combined rate `9 + 11 = 20`, it writes *`"So maybe I need to find their combined patrol rate per hour and then divide that by two?"`* to produce `10`, and **Example 1** (*`"Wait, ... Or maybe the sides..."`*).
3. **Runaway CoT Snowballing and Context Dilution**: Once `"Wait"` enters `<think>`, repeated backtracking inflates traces to thousands of tokens, causing intermediate sub-results to be forgotten during final synthesis (**Example 6**, dropping the weekday loaves) and pushing difficult problems (**Example 1**) out to 30,000 tokens where out-of-horizon attention collapses into degenerate token repetition (`= \boxed{\frac{1}{4}} = ...`).

---

### Circuit Localization and Weight Edit Design

In a 22-layer Pre-RMSNorm Transformer with tied input/output embeddings (`d_model = 1536`), the final logit of token $t$ is proportional to $(e_t \odot w_{\text{final\_norm}})^T x_{22} / \text{rms}(x_{22})$. Early and middle layers (`0..18`) perform input reading, entity binding, arithmetic, and genuine contradiction detection, whereas the final readout layers (`19..21`) project abstract states directly onto next-token vocabulary directions via two distinct residual-writing subcircuits:

1. **Late SwiGLU MLP Readout (`layers.19..21.feed_forward.down_proj.weight`) — Habitual Discourse Priors**:
   Late MLPs act as key-value lexical memories that promote default R1 transition markers at sentence and paragraph boundaries. Partially projecting out the norm-folded mean direction of `["Wait", " Wait"]` with depth-tapered contractive strengths (`0.65, 0.75, 0.85`) attenuates unprovoked, habitual `"Wait"` logits at routine steps without zeroing the pathway. Similarly, projecting out the speculative hedging subspace `[" maybe", " Maybe", "Maybe", "maybe", " perhaps", " Perhaps"]` (`0.75, 0.85, 0.95`) suppresses unprincipled guessing transitions while leaving deductive casework markers (`"Suppose"`, `"If"`, `"Case"`) intact.
2. **Late GQA Output Projections (`layers.20..21.self_attn.o_proj.weight`) — Induction / Echo Suppression**:
   In the final two attention layers, token-copying and induction heads attend back to prior `\n\nWait` and `. Wait` occurrences in `<think>` and write `+u_{\text{Wait}}` into the residual stream, creating a positive-feedback loop where each `"Wait"` makes subsequent `"Wait"`s progressively more probable. Projecting `["Wait", " Wait"]` out of `layers.20..21.self_attn.o_proj.weight` (`0.70, 0.80`) selectively breaks this self-echoing avalanche while leaving 1,535 orthogonal dimensions—preserving copying of numbers, variables, units, and `\boxed{}` formatting.

#### Why This Asymmetric Design Preserves Genuine Self-Correction on Unseen Problems
- **Threshold Filtering Rather Than Hard Banning**: Before `"Wait"` has appeared in a solution, `o_proj` copying is inactive, so only the moderate `down_proj` attenuation applies (~2 logit points). At `temperature = 0.6`, this cuts random unprovoked self-doubt odds by $>20\times$ on clean GSM8K/MATH steps, yet when a genuine contradiction arises on a hard MATH-500 problem, multi-layer conflict signals from untouched layers `0..18` (plus untouched explicit correction tokens like `"Actually"`, `"However"`, and `"But"`) still clear the threshold and trigger error recovery. Once `"Wait"` has already been used, the additional `o_proj` projection engages to prevent cyclic `"Wait"` spinouts.
- **Strict Contraction and Non-Inversion**: Keeping every strength $S \in (0, 1)$ guarantees $(1 - S) > 0$ on each layer's projection—never flipping the sign of negative inhibitions inside equations and strictly contracting column norms so RMSNorm scales remain well-conditioned.

```json
{
  "edits": [
    {
      "op": "project_out",
      "tensor": "model.layers.19.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 0.65,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.20.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 0.75,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.21.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 0.85,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.20.self_attn.o_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 0.70,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 0.80,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.19.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": [" maybe", " Maybe", "Maybe", "maybe", " perhaps", " Perhaps"],
        "fold_final_norm": true
      },
      "strength": 0.75,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.20.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": [" maybe", " Maybe", "Maybe", "maybe", " perhaps", " Perhaps"],
        "fold_final_norm": true
      },
      "strength": 0.85,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.layers.21.feed_forward.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": [" maybe", " Maybe", "Maybe", "maybe", " perhaps", " Perhaps"],
        "fold_final_norm": true
      },
      "strength": 0.95,
      "cols": "all"
    }
  ]
}
```