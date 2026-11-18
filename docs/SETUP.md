# Setup

This is the one-time setup of a machine, in the order it has to be done. The tested machine is
Windows 11 with WSL2 (Ubuntu), an NVIDIA GPU with 16 GB, and 64 GB of memory given to WSL. On
native Linux, "the Windows side" and "the Linux side" below are the same machine and every command
runs in bash; that arrangement has a code path but has not been tested.

Two terms are used throughout:

- **The driver side** is the Python that runs `scripts/`, `tests/` and
  `tools/prepare_tiny_models.py`. On the tested machine it is Windows Python, started from
  PowerShell in the repository root.
- **The Linux side** is a bash shell inside WSL. vLLM, the tools that run a tiny model, and the
  `oneshot_code` programs run there. The driver starts them by itself through `wsl.exe`; only the
  setup below and the optional tools are typed into a Linux shell by hand.

What each part of the benchmark needs:

| to run | you need |
|---|---|
| the self-test without tiny models | step 1 |
| `oneshot_spec` sessions | steps 1, 4, 5 and an API key (step 6) |
| `oneshot_code` sessions | the above and steps 2, 3b |
| scoring a tiny model | steps 1, 2, 3a, 4, 5 and, for the two math sets, `OPENAI_API_KEY` |
| the figures | step 1 and finished scoring |

## 1. The driver side

Python 3.10 or later (3.11 was used). From the repository root:

```
pip install -r requirements.txt
python tests/selftest.py
```

At this point the self-test checks everything that needs no tiny model and reports each tiny model
as NOT CHECKED. It should end without a failed check. (This model-free run was not exercised
during the port of the code; the full run with all five tiny models present was.)

## 2. WSL2 and the GPU

Install WSL2 with Ubuntu (`wsl --install` in an administrator PowerShell) and a current NVIDIA
driver on Windows. In the Linux shell, `nvidia-smi` should list the GPU.

By default WSL is given part of the machine's memory. The `oneshot_code` programs are allowed up
to 56 GiB (see `WSPMB_SANDBOX_MEM_GB` in step 4), so the memory given to WSL in
`%UserProfile%\.wslconfig` should be above that, or the limit lowered.

## 3. The two Linux virtual environments

Both are created in the Linux shell. The default locations are the ones below; other locations
work if `.env` names them (step 4).

**3a. The vLLM venv** serves the tiny models on the GPU and runs the practice-run and
checkpoint-check tools. The benchmark was run with vLLM 0.29.0 and math-verify 0.9.0:

```bash
python3 -m venv ~/venvs/vllm
~/venvs/vllm/bin/pip install vllm==0.29.0 math-verify==0.9.0
```

vLLM brings its own PyTorch and transformers. If the plain `pip` route fails on your system,
follow vLLM's own installation instructions for that version; the benchmark only needs the
`vllm` command and Python package to exist in this venv. Check:

```bash
~/venvs/vllm/bin/python -c "import vllm, math_verify; print(vllm.__version__)"
```

**3b. The sandbox venv** runs the programs of the `oneshot_code` tier, on the CPU. The prompt tells
the primary LLM that numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, safetensors and
tokenizers are available, so install all of them:

```bash
python3 -m venv ~/venvs/wsp_sandbox
~/venvs/wsp_sandbox/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venvs/wsp_sandbox/bin/pip install numpy scipy scikit-learn pandas sympy safetensors tokenizers
```

Check:

```bash
~/venvs/wsp_sandbox/bin/python -c "import torch, numpy, scipy, sklearn, pandas, sympy, safetensors, tokenizers; print('ok')"
```

## 4. Settings: the `.env` file

Copy `.env.example` to `.env` in the repository root and fill it in. `wspmathbench/config.py`
reads this file on both sides. Put the settings in `.env` and not only in the Windows
environment: the Linux-side tools load `config.py` too, and they see `.env` but not Windows
environment variables. A variable that is set in the environment wins over the file.

| variable | default | meaning |
|---|---|---|
| `WSPMB_MODELS_DIR` | `<home>/wspmathbench_models` | Folder with the tiny models and, under `wsp_checkpoints/`, the edited checkpoints (up to about 2 GB each). On Windows, a Windows path on a local disk; the Linux side reads it as `/mnt/<drive>/...`. Set this one explicitly: the home folder differs between Windows and WSL, so the default does not name the same folder on both sides. |
| `WSPMB_HF_CACHE_DIR` | `HF_HUB_CACHE` if set, otherwise `<home>/.cache/huggingface/hub` | The Hugging Face cache that `tools/prepare_tiny_models.py` copies the models out of. |
| `WSPMB_RESULTS_DIR` | `<repository>/results` | Where sessions, scores, figures and error logs are written. |
| `WSPMB_VLLM_VENV` | `~/venvs/vllm` | The vLLM venv (a Linux path). |
| `WSPMB_SANDBOX_PYTHON` | `~/venvs/wsp_sandbox/bin/python` | The Python of the sandbox venv (a Linux path). |
| `WSPMB_SERVER_LOG_DIR` | `~/wspmathbench_server_logs` | Linux folder for the console output of the tiny-model server, one file per launch. |
| `WSPMB_SERVER_PORT` | `8000` | Port of the tiny-model server. |
| `WSPMB_SANDBOX_MEM_GB` | `56` | Memory limit of a `oneshot_code` program, in GiB. The prompt quotes it, so another value changes what the primary LLM is told; the paper's sessions ran with 56. |
| `WSPMB_SANDBOX_THREADS` | `8` | CPU threads of a `oneshot_code` program. |

The time limit of a `oneshot_code` program (900 seconds, also quoted in the prompt) is set in
`config.SANDBOX` and has no environment variable.

Check, on the driver side:

```
python -c "from wspmathbench import config as c; print(c.MODELS_DIR, c.RESULTS_DIR, c.SANDBOX)"
```

## 5. The tiny models

The benchmark pins each tiny model to one revision (`config.TINY_MODEL_REGISTRY`) and reads it
from a plain folder under the models folder:

| tiny model | repository | revision | folder under the models folder |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | `Qwen3.5-0.8B` |
| MobileLLM-R1-950M | `facebook/MobileLLM-R1-950M` | `aa7d61df54d764738d32c220e107ed5f292d56eb` | `MobileLLM-R1-950M-bf16` |
| LFM2.5-350M | `LiquidAI/LFM2.5-350M` | `9921cd4d82c0e3986887aa4b376f9446729ef8a3` | `LFM2.5-350M` |
| gemma-3-270m-it | `google/gemma-3-270m-it` | `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` | `gemma-3-270m-it` |
| TinyStories-33M | `roneneldan/TinyStories-33M` | `2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8` | `TinyStories-33M-bf16` |

Downloading needs the Hugging Face command-line tool on the driver side
(`pip install -U huggingface_hub`). Some of these repositories ask you to accept a licence on
their Hugging Face page and to log in (`hf auth login`) first.

**Qwen3.5-0.8B** is downloaded straight into its folder as plain files:

```
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir <models folder>/Qwen3.5-0.8B
```

**The other four** are downloaded into the Hugging Face cache and then copied out of it by
`tools/prepare_tiny_models.py`, which also converts two of them to bfloat16 (every tiny model runs
in bfloat16, and the benchmark's starting weights must be the numbers the running model uses):

```
hf download facebook/MobileLLM-R1-950M --revision aa7d61df54d764738d32c220e107ed5f292d56eb
hf download LiquidAI/LFM2.5-350M --revision 9921cd4d82c0e3986887aa4b376f9446729ef8a3
hf download google/gemma-3-270m-it --revision ac82b4e820549b854eebf28ce6dedaf9fdfa17b3
hf download roneneldan/TinyStories-33M --revision 2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8
python tools/prepare_tiny_models.py
```

`prepare_tiny_models.py` leaves an existing folder alone unless `--force` is given, and takes
model names to prepare only some (`python tools/prepare_tiny_models.py TinyStories-33M`). For a
converted model it writes `wsp_source.json` into the folder, with the source revision, the source
file's SHA-256 and what was done.

The download commands above were not run while this document was written; the revisions and
folder names are those in `config.py`.

Check, on the driver side. With the models in place the self-test checks each of them, including
stub sessions; the checks that start the `oneshot_code` kernel need the sandbox venv of step 3b
(`--skip-kernel-limits` leaves most of them out):

```
python tests/selftest.py
python scripts/run_sessions.py --provider stub --model stub-spec --tiny-model TinyStories-33M --smoketest
```

## 6. API keys

Add to `.env` the keys of the providers you will use:

| key | needed for |
|---|---|
| `OPENAI_API_KEY` | gpt-6.1-sol, gpt-6-astra, gpt-6-astra-pro; and scoring any tiny model on GSM8K-500 or MATH-500 (the answer helper) |
| `ANTHROPIC_API_KEY` | claude-fable-5-2 |
| `OPENROUTER_API_KEY` | the presets of gemini-4-carbon, deepseek-v4.1-flash-1017 and kimi-k3.1-pro, which go through OpenRouter |
| `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY` | only when one of those three models is called on its provider's own API, with `--llm-kwargs '{"host": null}'` |

The self-test and the stub provider need no key.

## 7. A first scoring run

Scoring starts the tiny model's server on the Linux side, in a new console window on Windows,
waits for it to answer, and stops whatever other model was on the port. Nothing else should be
using the GPU. A trial that asks only the first question of the set and saves nothing:

```
python scripts/score_tiny_model.py --tiny-model TinyStories-33M --smoketest
```

TinyStories-33M is served by `tools/serve_transformers.py` in the vLLM venv, because vLLM 0.29.0
cannot load it; the four chat models are served by `vllm serve`. BasicArithmetic is graded by rule,
so this trial needs no API key. The same command with `--tiny-model Qwen3.5-0.8B` tries the vLLM
route and needs `OPENAI_API_KEY`.

If a server does not come up, its console output is in the folder named by
`WSPMB_SERVER_LOG_DIR`; `wspmathbench.llm.query.local_server_log()` prints the end of the latest
log.

Launching and stopping the servers was not exercised during the port of the code to this
repository (it was used throughout the paper's runs in the code this was ported from), so treat
the first scoring run as the test of that step.

## Practice runs

Nothing needs to be done: the practice runs the paper's sessions used are in
`data/practice_runs/`. To produce one yourself (for a new tiny model, or to reproduce the shipped
ones), in the Linux shell at the repository root, with the GPU free:

```bash
~/venvs/vllm/bin/python tools/run_practice_baseline.py --tiny-model gemma-3-270m-it --run-name my_run
```

It answers 128 practice problems from each source of the task (`--n-per-source 0` for the whole
pool, as for TinyStories-33M's `baseline_1000`) and writes
`data/practice_runs/<tiny model>/<protocol>/<run name>/`, at the very end and only if every
generation finished. An existing run is not overwritten unless `--overwrite` is given.
