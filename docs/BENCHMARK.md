# The benchmark in full

This document states the rules of WSPMathBench as the code implements them. File and function
names refer to `wspmathbench/` unless a folder is given.

Terms: the **primary LLM** is the model being evaluated. A **tiny model** is one of the five small
language models whose weights it edits. A **tier** is the form its reply must take. A **session**
is one prompt and one reply for one tiny model in one tier. A **practice run** is the unedited
tiny model's answers to practice problems. A **checkpoint** is a full copy of a tiny model with a
session's edit applied.

## 1. A session, step by step

`benchmark.WSP_Benchmark` runs one session; `scripts/run_sessions.py` is its command line. The
steps are numbered as in `session.py`.

1. **Open the tiny model** (`tiny_model.py`). Tensor names, shapes, layer kinds and the number of
   heads are read from the model's files, so nothing about a model is typed by hand elsewhere.
2. **Build the diagnostic pack** (`diagnostics.py`) from the practice run: the accuracy overall
   and by source, for the math task the mean answer length and the share of answers cut at the
   token limit, the counts of each kind of failure, and 8 failing answers chosen at random with a
   fixed seed. A failing math answer is shown as its first 600 and last 400 characters next to the
   problem's reference answer and solution; a BasicArithmetic answer is shown whole. The pack
   states what happened and never why.
3. **Render the prompt** (`prompts.py`) from the template of the tier and prompt version, filled
   with the architecture, the table of editable tensors, the task and protocol sentences, the
   pack, and either the description of the edit language or the description of what a program
   has available. The rendered prompt is saved as `prompt.txt`.
4. **Query the primary LLM and use its reply.** The reply must contain exactly one fenced block
   of the tier's kind: ` ```json ` for `oneshot_spec`, ` ```python ` for `oneshot_code`. Fenced
   blocks of other kinds in the reasoning do not count.
   - If there is no such block, or more than one, or a `oneshot_spec` edit breaks a rule of the
     edit language, the primary LLM is sent the original prompt again together with its reply and
     the list of what was wrong. There are at most 3 attempts in all.
   - A `oneshot_spec` edit that passes is applied by the benchmark.
   - A `oneshot_code` program is run once. A program that raises an exception or runs out of
     time is not sent back: that is an outcome of the session, and the edits it saved before
     stopping are kept.
   - An API failure is retried separately and does not use up an attempt.
5. **Write the edit manifest** (`edits.manifest`): for each edited tensor its name, shape, dtype,
   the number of entries changed, the relative size of the change, and the SHA-256 of the original
   and of the edited tensor.
6. **Set the status** (section 5).
7. **Build the checkpoint** (`checkpoint.py`) if an edit was saved: the tiny model's files, a new
   `model.safetensors` with the edited tensors cast to the model's dtype, and `edits.json`, which
   holds the manifest, the tiny model, its revision, the task and the session ID. The folder is
   `<models folder>/wsp_checkpoints/<session ID>/` and loads like the unedited model.
8. **Append the session's row** to `Master_WSP_Edits.csv`.

A failure inside a session is recorded as its outcome, with whatever cost was already spent.
Wrong arguments raise before a session ID is used or any money is spent.

## 2. The edit rules (both tiers)

`edits.save_edit` enforces them for the edit language and for programs alike:

- Only the tensors of the language model can be edited, and of those not the frozen ones: the
  input embedding, the final norm and the output layer (and, for TinyStories-33M, the final norm's
  bias and the position embedding).
- A tensor keeps its shape, and every value must be finite.
- An edit that only multiplies a whole tensor by a constant is refused.
- An edit equal to the original values is not an edit: nothing is saved for that tensor.

Edits are stored as float32 `.npy` files in the session's `edited/` folder, one per tensor.

## 3. The `oneshot_spec` edit language

The reply's JSON block is a list of operations that are applied together (`dsl.py`). `NAME` is an
editable tensor from the prompt's table.

| operation | effect |
|---|---|
| `{"op": "zero_head", "layer": L, "head": H}` | zeroes the columns of layer L's attention output projection that read head H. Only for layers whose attention output is split into heads. |
| `{"op": "zero_columns", "tensor": NAME, "cols": [...]}` | sets those columns to zero |
| `{"op": "zero_rows", "tensor": NAME, "rows": [...]}` | sets those rows to zero |
| `{"op": "scale_columns", "tensor": NAME, "cols": [...], "factor": F}` | multiplies those columns by F |
| `{"op": "scale_rows", "tensor": NAME, "rows": [...], "factor": F}` | multiplies those rows by F |
| `{"op": "project_out", "tensor": NAME, "direction": DIR, "strength": S, "cols": [...] or "all"}` | removes S times the component along a unit direction u from the chosen columns: `W[:, cols] = W[:, cols] - S * u (u^T W[:, cols])`. Only for tensors that write to the residual stream (and, for TinyStories-33M, the bias vectors added with them, with `"cols": "all"`). |

A direction `DIR` is resolved from the weights, without running the model:

| direction | meaning |
|---|---|
| `{"type": "unembed_token", "token": "the", "fold_final_norm": true}` | the output-embedding row of a token, which must encode to exactly one token id. With `fold_final_norm` the row is multiplied elementwise by the final norm's weight (its gain, 1 + weight, for gemma-3-270m-it). |
| `{"type": "unembed_mean", "tokens": ["the", " the"], "fold_final_norm": true}` | the mean of several such rows |
| `{"type": "svd", "tensor": NAME, "component": K}` | the K-th left singular vector of a residual-writing tensor |

Every direction is normalised to unit length. In gemma-3-270m-it a norm sits between each
residual-writing tensor and the residual stream; there an `unembed_token` or `unembed_mean`
direction is also multiplied by that norm's gain, unless the direction carries
`"fold_post_norm": false`.

Limits (`config.py`): at most 8 operations; at most 64 indices per operation (`"all"` only in
`project_out`); factors and strengths between 0 and 4; indices are zero-based.

The exact wording shown to the primary LLM is produced by `dsl.render_dsl_text` from the same
description of the model that the validator reads, so the prompt and the validator cannot
disagree.

## 4. A `oneshot_code` program

The program runs once on the Linux side, on the CPU, as `kernel_runner.py` in the sandbox venv
(`kernel.py` starts it). It has, defined in advance:

| name | meaning |
|---|---|
| `NAMES` | the editable tensor names |
| `SHAPES` | name to shape |
| `load(name)` | the tensor as a float32 torch tensor, with its original values |
| `save_edit(name, t)` | replace that tensor with `t`; refused under the rules of section 2 |
| `drop_edit(name)` | undo a `save_edit` |
| `edited()` | the names saved so far |
| `tok` | the tokenizer: `tok.encode(text)`, `tok.decode(ids)` |

It can also read the model's files and the practice problems, and write in its work folder.
numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, safetensors and tokenizers are
installed. Gradients are disabled.

Limits (`config.SANDBOX`): 900 seconds; 56 GiB of memory by default; 8 CPU threads by default; no
GPU; the first 10,000 characters of printed output are kept. The program does not see its own
output, and the primary LLM is not shown it either.

The prompt forbids three things: gradient descent or any other training; running the model, or
implementing its forward computation, on any input; and internet access. These are rules of the
task stated in the prompt. They are not enforced by the code, and the program is not isolated
from the machine (see Safety in the README).

## 5. How a session can end

| status | meaning | checkpoint |
|---|---|---|
| `edit_saved` | at least one tensor was edited. This includes a program that raised after saving edits. | built |
| `empty_edit` | the reply was accepted and applied or run, and no tensor changed. For a program this covers both a program that raised before saving anything and one that finished without saving. | none |
| `no_valid_reply` | no usable reply within 3 attempts | none |
| `failed` | the API call failed after its retries, or an error occurred inside the session | none |

Scoring (`llm/scoring.score_tiny_model`) treats a session with no edit as follows:

- `empty_edit` where the program raised: no model is run. Rows with score 0 are written for each
  evaluation set, every question recorded as unanswered. This is the paper's rule that an edit
  that raises an error counts as a tiny-model score of 0.
- `empty_edit` where the reply ran without error and changed nothing: the unedited model is run
  and its results are recorded under the session.
- `no_valid_reply` and `failed` sessions have no checkpoint and cannot be scored. Running the
  session again replaces it, because the latest session of a primary LLM for a tiny model and
  tier is the one that counts (section 8).

## 6. Prompt versions

A prompt version names the scope of the prompt. `v1` is written around repairing a fault and is
the default for Qwen3.5-0.8B; `v2` is open-ended and is the default for the other four tiny
models. There is one template per version and tier in `prompts/`, filled per tiny model, and a
row records the version with the model's name appended (for example `v2-gemma-3-270m-it`).

The prompt is part of what is being measured, so a template that sessions have used is meant
never to change in place: a change is a new version and a new file. One such in-place change did
happen to the v2 templates during the paper's runs; the README lists it.

The prompt is deliberately free of hints. It gives measurements and failing answers, and does not
say what is wrong with a model or what to look for. The example tokens in the description of the
edit language are ordinary words for the same reason.

## 7. Evaluation protocols and grading

Each tiny model has one protocol (`config.PROTOCOLS`), taken from its authors' recommended
settings. Every tiny model runs in bfloat16.

| tiny model | format | sampling | new-token limit |
|---|---|---|---|
| Qwen3.5-0.8B | chat template, thinking enabled | temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 | 32,768 |
| MobileLLM-R1-950M | chat template | temperature 0.6, top_p 0.95 | 30,000 |
| LFM2.5-350M | chat template | temperature 0.1, top_k 50, repetition penalty 1.05 | 30,000 |
| gemma-3-270m-it | chat template | temperature 1.0, top_k 64, top_p 0.95 | 30,000 |
| TinyStories-33M | plain completion of the question text | temperature 0.1 | 16 |

The four chat models are asked: "Please reason step by step, and put your final answer within
\boxed{}." In the practice runs, and in the protocol text the prompt quotes, that instruction is
the system message for MobileLLM-R1-950M and follows the problem in the user message for the
other three. When a tiny model is scored (`llm/benchmarks.py`), the instruction is sent as the
system message for all four. TinyStories-33M gets the question text and nothing else.

Grading, per evaluation set (`llm/benchmarks.eval_benchmark_async`):

- **GSM8K-500.** A reply that is nothing but digits is the answer; an empty reply is graded 0.
  Any other reply goes to the answer helper, an OpenAI model (gpt-5.6-luna at low effort), which
  returns the integer inside the last `\boxed{}`, or 0 if there is none, if the answer is not
  inside `\boxed{}`, or if the text is incomplete or unclear. The integer is compared with the
  reference answer. No reference answer is 0.
- **MATH-500.** The content of the last `\boxed{}` is read by a fixed rule; only a reply with no
  readable `\boxed{}` goes to the answer helper, which returns the boxed answer or `NA`. The
  answer is compared with the reference by math-verify 0.9.0 for mathematical equivalence, with
  10 seconds per answer.
- **BasicArithmetic.** The first number in the continuation, written in digits or in words, is
  compared with the reference as an integer (`arithmetic.first_number`). No helper is called.

A reply cut at the token limit before any answer text is graded as unanswered. A question that
fails every attempt is recorded as unanswered.

## 8. The score

Computed by `analysis.py`; `scripts/make_figures.py` runs it.

**Which session counts.** A configuration is a primary LLM at one reasoning effort, named
`<model> (<effort>)`. When it has several sessions for the same tiny model and tier, the one with
the largest ID counts, whatever its status. A configuration gets a score only when it has a
session for every tiny model in the tier and all of them, and the unedited models, have been
scored.

**One comparison.** For a tiny model and one of its evaluation sets, each question's score for a
model is the fraction of that model's scoring passes that answered it correctly. The change is
the mean over questions of the edited model's score minus the unedited model's. Its standard
error is the paired one: the standard deviation of the per-question differences divided by the
square root of the number of questions. The difference is tested with a two-sided z-test, with
the Holm-Bonferroni correction at the 95% family-wise level (`--confidence` changes the level).

**The score.** A change counts as it is when it is significant and as zero when it is not; a
significant loss counts with its negative sign. The counted changes are averaged over the
evaluation sets of a tiny model, so that a tiny model with one evaluation set weighs as much as
one with two, and then over the tiny models. Scores are fractions in the code and are reported
in percentage points.

**Its standard error.** With the evaluation sets taken as independent:

```
SE(tiny model) = sqrt(sum of SE^2 over its evaluation sets) / number of its evaluation sets
SE(score)      = sqrt(sum of SE(tiny model)^2 over the tiny models) / number of tiny models
```

A change that counts as zero keeps its standard error. The standard error covers the sampling of
evaluation questions. It does not cover the variation between repeated sessions of the same
primary LLM, since each comparison rests on one session.

The bar figures compare the unedited tiny model with the models edited by every primary LLM on
one evaluation set. There every pair of bars is tested and corrected together, and bars that
share a letter do not differ significantly.

## 9. Files

**A session** writes `results/sessions/<ID>_<tier>_<tiny model>_<primary LLM>_<effort>/`
(`results/smoketest/` for a smoketest):

| file | content |
|---|---|
| `prompt.txt` | the rendered prompt |
| `turns/t001/answer.md`, `reasoning.md`, `run.json` | the reply, the reasoning text if the provider returned any, and the call's settings, tokens, cost and time. A further attempt is `t002`, `t003`. |
| `tool_calls.jsonl` | what was applied: the edit specification, or the program with its printed output, run time and exception |
| `work/` | `oneshot_code` only: the program, its arguments, its output |
| `edited/<tensor name>.npy` | the edited tensors, float32 |
| `edit/manifest.json` | the edit manifest |
| `state.json` | the session's settings, status, cost, time, edited tensors and checkpoint folder |

**`results/Master_WSP_Edits.csv`** has one row per session: `ID`, `date_time`,
`benchmark_version`, `prompt_version`, `dsl_version`, `tier`, `tiny_model`, `tiny_revision`,
`task`, `provider`, `model`, `llm_kwargs`, `status`, `n_attempts`, `tensors_edited`,
`entries_changed`, `relative_change`, `total_cost`, `token_list`, `wall_s`, `session_dir`,
`checkpoint_dir`, `misc`. Sessions that run at the same time can add their rows out of order, so
sort by `ID`. Only one Python process at a time should write to the results folder.

**A scoring pass** appends one row to `results/scores/Master_LLM_<evaluation set>_Results.csv`:
`ID`, `date_time`, `provider` (`local`), `model` (`wsp-<tiny model>-base` for the unedited model,
`wsp-<tiny model>-<session ID>` for an edited one), `llm_kwargs`, `score`, `model_answers` (the
graded answers in file order), `total_cost`, `cost_list`, `answer_cost_list`, `token_list` and
`misc` (the grading rule, notes, and under `wsp` the session ID, tiny model, task and protocol).
The full replies go to
`results/scores/<evaluation set>_model_answers/<row ID>_<evaluation set>_model_answers.txt`.

**The analysis** writes `results/compiled_wsp_benchmark_metrics.parquet` (one row per primary LLM
and tier: `model`, `tier`, `score`, `score_se`, and the counted change and its standard error per
tiny model) and the figures in `results/figures/`.
