# WSPMathBench

WSPMathBench measures whether a large language model (LLM) can improve a small LLM's math abilities by editing its
weights directly. The large model (the **primary LLM**) is shown a description of a small language
model (the **tiny model**): its architecture, the names and shapes of its tensors, and how it
performed on practice problems. It replies once, with an edit to the tiny model's weights. It never
runs the tiny model, never trains it, and never sees the result. The edited tiny model is then
scored on held-out mathematics questions, and the primary LLM's score is how much its edits changed
those results. This is weight-space programming (WSP), after Takehi et al., *Sparks of Weight
Space Programming: LLMs can understand and write neural network weights*.

This repository holds the code that runs the benchmark, the prompts, the data, and a few example
sessions. 

## How it works

A **session** is one prompt to a primary LLM and its one reply, for one tiny model in one tier.

There are two **tiers**, which differ in the form of the reply:

| tier | the reply contains | how the edit is applied |
|---|---|---|
| `oneshot_spec` | one JSON block in a small edit language (zero or scale rows, columns or attention heads; project a direction out of a tensor) | the benchmark carries out the operations |
| `oneshot_code` | one Python program | the program runs once, on the CPU, with a time and memory limit, and replaces tensors through `save_edit` |

In both tiers the input embedding, the final norm and the output layer are frozen, an edit that
only multiplies a whole tensor by a constant is refused, and every value must be finite.

There are five tiny models, each with a fixed evaluation protocol:

| tiny model | Hugging Face repository | evaluation sets | prompt version |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | GSM8K-500, MATH-500 | v1 |
| MobileLLM-R1-950M | `facebook/MobileLLM-R1-950M` | GSM8K-500, MATH-500 | v2 |
| LFM2.5-350M | `LiquidAI/LFM2.5-350M` | GSM8K-500, MATH-500 | v2 |
| gemma-3-270m-it | `google/gemma-3-270m-it` | GSM8K-500, MATH-500 | v2 |
| TinyStories-33M | `roneneldan/TinyStories-33M` | BasicArithmetic | v2 |

GSM8K-500 is 500 questions from the GSM8K test set, MATH-500 is the 500-question subset of the
MATH test set, and BasicArithmetic is a set of 500 very easy arithmetic questions built for this
benchmark (`tools/make_basic_arithmetic.py`). Prompt version v1 is written around repairing a
fault; v2 is open-ended.

What the primary LLM is told about the tiny model's behaviour comes from a **practice run**: the
unedited tiny model's answers to practice problems (GSM8K and MATH training problems, or
BasicArithmetic practice questions), produced with the same protocol that is used later for the final evaluation. The
prompt quotes the totals of that run and eight incorrect answers chosen at random. The practice runs
the paper used are in `data/practice_runs/`.

After a session, the edit is written into a full copy of the tiny model, the **checkpoint**, which
is what gets scored.

**The score.** Each tiny model, unedited and edited, answers its evaluation sets several times
(three passes by default). For each tiny model and evaluation set, the change in score is the mean
over questions of the edited model's score minus the unedited model's, with a paired per-question
standard error. The changes are tested at the 95% level with the Holm-Bonferroni method; a
change that is not significant counts as zero, and a significant loss is counted as such. The counted changes are averaged over the evaluation sets of each tiny model and then over
the five tiny models. Scores are reported in percentage points and can be negative. A perfect
score, which would require every edited tiny model answering every evaluation question correctly, is roughly 77 points.
`docs/BENCHMARK.md` gives the rules in full.

## Repository layout

```
wspmathbench/          the benchmark
  config.py              paths, limits, tasks, protocols, the tiny-model registry
  benchmark.py           one session from arguments to a row of results
  session.py             the steps of a session
  tiny_model.py, diagnostics.py, prompts.py     reading the model, the practice summary, the prompt
  dsl.py, edits.py       the oneshot_spec edit language; the edit rules and the edit manifest
  kernel.py, kernel_runner.py                   running a oneshot_code program
  checkpoint.py          building the edited checkpoint
  stub.py                canned replies for trial runs that cost nothing
  arithmetic.py          reading a number out of a BasicArithmetic answer
  analysis.py            statistics, the score, figures
  llm/                   provider clients, the model registry, the local tiny-model server,
                         scoring a tiny model on an evaluation set, the MATH-500 grader
scripts/               the three entry points: run_sessions.py, score_tiny_model.py, make_figures.py
tools/                 one-time and Linux-side tools: preparing the tiny models, practice runs,
                       the checkpoint check, the server for TinyStories-33M, the BasicArithmetic generator
prompts/               the prompt templates (two tiers, two versions, and the system prompts)
data/eval/             the three evaluation sets
data/practice/         the practice pools
data/practice_runs/    the unedited tiny models' practice answers, as used by the paper's sessions
examples/              nine of the paper's sessions: prompt, reply, and for some the full reasoning
tests/selftest.py      an end-to-end check that needs no API key and no GPU
docs/                  SETUP.md, BENCHMARK.md, ADDING_MODELS.md
```

## Requirements

The tested setup is Windows 11 with WSL2 (Ubuntu) and one NVIDIA GPU; the runs in the paper used a
16 GB GPU and 64 GB of memory given to WSL. The sessions, the analysis and the scripts run in
Windows Python; the tiny models are served by vLLM, and the `oneshot_code` programs run, inside
WSL.

On native Linux the same commands are meant to run directly in bash (`config.shell_command`, and
the server launch in `llm/query.py`). That path has not been tested.

Three Python environments are involved (`docs/SETUP.md` builds them):

| environment | where | used for |
|---|---|---|
| the driver, `requirements.txt` | Windows (Python 3.10 or later; 3.11 was used) | sessions, scoring, analysis, the self-test |
| the vLLM venv, vLLM 0.29.0 | Linux side | serving and running the tiny models on the GPU |
| the sandbox venv | Linux side | the `oneshot_code` programs, CPU only |

## Quick start (no API key, no GPU, no cost)

All commands are run from the repository root.

```
pip install -r requirements.txt
python tests/selftest.py
```

Without any tiny model on disk the self-test checks reply parsing, the prompt templates,
BasicArithmetic and the pure functions of the practice-run tool, and reports each tiny model as
NOT CHECKED. Once the tiny models are prepared (`docs/SETUP.md`), the same command checks each of
them, including sessions with the stub provider and, when the sandbox venv exists, the
`oneshot_code` kernel.

A trial session with the stub provider runs the whole path of a session with a canned reply. It
needs the tiny model's folder but no API key. It writes into `results/smoketest/` and
`<models folder>/wsp_checkpoints/smoketest/` (both cleared on the next smoketest) and adds no row:

```
python scripts/run_sessions.py --provider stub --model stub-spec --tiny-model TinyStories-33M --smoketest
```

## The full pipeline

| step | command | needs | cost |
|---|---|---|---|
| 1. Set up the machine and prepare the tiny models | `docs/SETUP.md`, then `python tools/prepare_tiny_models.py` | downloads from Hugging Face | free |
| 2. Practice runs | already in `data/practice_runs/`; nothing to do | | |
| 3. Run edit sessions | `python scripts/run_sessions.py --model gpt-6-astra --tiny-model gemma-3-270m-it --tier oneshot_spec` | an API key; the sandbox venv for `oneshot_code` | the primary LLM's API price |
| 4. Score the unedited tiny model, once per tiny model | `python scripts/score_tiny_model.py --tiny-model gemma-3-270m-it` | the GPU, `OPENAI_API_KEY` for the two math sets | a small OpenAI cost |
| 5. Score each session's checkpoint | `python scripts/score_tiny_model.py --session-id 12` | the GPU, `OPENAI_API_KEY` for the two math sets | a small OpenAI cost |
| 6. Compile the scores and draw the figures | `python scripts/make_figures.py` | | free |

Notes on the steps:

- **Step 3.** For one of the paper's seven primary LLMs, `--model` alone selects the provider and
  the settings the paper used (`PRESETS` in `scripts/run_sessions.py`). A full evaluation of one
  primary LLM is ten sessions: each of the five tiny models in each of the two tiers. In the
  paper's runs those ten sessions cost between $0.26 and $19.72 per primary LLM. Each session
  prints its ID; the edited checkpoint is written to `<models folder>/wsp_checkpoints/<ID>/`.
- **Steps 4 and 5.** Scoring starts the tiny model's server on the Linux side by itself, stopping
  whatever other model is on the port, so the GPU must be free and only one scoring run can be
  active. The default is three passes over each evaluation set (`--repeats`). An edited model and
  its unedited model must be scored with the same `--max-tokens`. `--smoketest` asks only the
  first question of each set and saves nothing.
- **Step 6.** A primary LLM gets a score once it has a session for every tiny model in both tiers
  and all of them, and the five unedited models, are scored. When a configuration has several
  sessions for the same tiny model and tier, the latest one counts. If anything is missing it is shown
  with a print statement.

Optional: `checkpoint.check_in_vllm(<session ID>)`, or `tools/check_checkpoint_vllm.py` on the
Linux side, loads a checkpoint and answers a few practice problems, as a check that an edited tiny model still
produces text. It is not scored.

## Where results are written

Everything a run writes goes under `results/` (setting `WSPMB_RESULTS_DIR`), which git ignores:

```
results/
  Master_WSP_Edits.csv                 one row per edit session
  sessions/<ID>_<tier>_<tiny model>_<primary LLM>_<effort>/
                                       the prompt, the reply, the edit and its manifest
  scores/Master_LLM_<evaluation set>_Results.csv
                                       one row per scoring pass of a tiny model
  scores/<evaluation set>_model_answers/
                                       the full answers of each pass
  compiled_wsp_benchmark_metrics.parquet   the scores, written by make_figures.py
  figures/                             the figures, as PDF and PNG
  error_logs/                          tracebacks of failed calls
```

Error logs contain absolute paths of the machine they were written on.

## Reproducing the paper

The paper evaluates seven primary LLMs, each with one session per tiny model and tier (70
sessions): gpt-6.1-sol, gpt-6-astra, gpt-6-astra-pro and kimi-k3.1-pro at reasoning effort max,
deepseek-v4.1-flash-1017 and gemini-4-carbon at high, and claude-fable-5-2 at medium. To repeat it,
run steps 3 to 6 above for each of them with the presets. A primary LLM's replies may differ from run
to run, so new sessions give new edits; the paper's scores rest on a true one-shot approach.

### Known differences from the paper's runs

1. **The v2 prompt templates changed once during the paper's runs.** One sentence was added to the
   WHAT COUNTS paragraph of `prompts/oneshot_spec_v2.txt` and `prompts/oneshot_code_v2.txt`, the
   sentence beginning "Nor is an edit that makes the model give the same answer", in anticipation 
   of an edge case where the primary LLM decides to make the tiny LLM produce a constant answer, 
   rather than have any dependency on the problem
2. **A few sessions used slightly different settings from the presets.** Three of the ten
   gemini-4-carbon sessions ran with `max_tokens` 128000, two of them on Google's own API instead
   of OpenRouter (a comment in `scripts/run_sessions.py` says so). One gpt-6.1-sol session ran
   without OpenAI's flex service tier.
3. **At scoring time the `\boxed{}` instruction is a system message for all four chat models.**
   The protocols in `config.py`, which the prompt quotes and the practice runs follow, put the
   instruction in the user message for Qwen3.5-0.8B, LFM2.5-350M and gemma-3-270m-it. The scoring
   code sends it as the system message for every chat model. The unedited and the edited model
   are scored the same way, so the comparison between them is unaffected.
4. **Grading GSM8K-500 and MATH-500 calls an OpenAI model by default.** A reply whose final answer cannot be
   read by a fixed rule goes to a small helper model (the paper employed gpt-5.6-luna at low effort), which returns
   the content of the last `\boxed{}`. Scoring those two sets therefore needs `OPENAI_API_KEY` and
   has a small cost. BasicArithmetic is graded purely deterministically. Modifications will be needed if you wish to use a different grader model. 

## Examples

`examples/` holds nine of the paper's sessions: the whole prompt, the whole reply, the program's
output for the `oneshot_code` tier, and the change in the tiny model's score. Four of them include
the primary LLM's full reasoning text. `examples/README.md` lists them.

## Safety

The `oneshot_code` tier runs a program written by an LLM on your machine. It runs as a separate
process in its own virtual environment, with a cleared environment, a time limit and a memory
limit. That is not a sandbox in the security sense: the program runs with your Linux user's
permissions, can read and write whatever that user can (under WSL this includes the Windows
drives, and so the `.env` file with the API keys), and has network access. The prompt forbids
training, running the tiny model and using the internet, but these are not enforced by the sandbox. Run
this tier only on a machine where that is acceptable, and consider reading the program in a reply
(`reply.md` in the examples, `turns/t001/answer.md` in a session folder) before trusting a
primary LLM you have not used before.

## Data credits and licences

- `data/practice/gsm8k_train.jsonl` and `data/practice/math_train.jsonl` are copied unchanged from
  [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs),
  `exp8_math/practice/`, at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, released under
  CC BY 4.0 by the authors of *Sparks of Weight Space Programming*. The problems themselves come
  from the GSM8K and MATH training sets and stay under those datasets' own licences.
- `data/eval/GSM8K_500_Benchmark.jsonl` and `data/eval/MATH500_Benchmark.jsonl` are subsets of
  the GSM8K and MATH test sets. Consult the original datasets for their licences.
- `data/eval/BasicArithmetic_Benchmark.jsonl` and `data/practice/basic_arithmetic_practice.jsonl`
  were generated for this benchmark by `tools/make_basic_arithmetic.py`.
- `tools/run_practice_baseline.py` adapts the answer extraction and checker call of
  `exp8_math/code/mathqa.py`, and `wspmathbench/checkpoint.py` adapts
  `exp8_math/code/build_ckpt.py`, both from the same repository under the MIT License.
- The tiny models are not distributed here. Each is downloaded from its Hugging Face repository
  under its own licence.

## Citation

arXiv submission pending.

## Licence

The code in this repository is released under the MIT License (`LICENSE`).
