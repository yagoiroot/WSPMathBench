# Adding models

Two kinds of model can be added: a primary LLM to evaluate, and a tiny model to be edited. The
first is a small change. The second changes the benchmark: the score is an average over the tiny
models, so scores computed with a different set of tiny models are not comparable with the
paper's.

## Adding a primary LLM

### A model of a provider that is already supported

The supported providers are OpenAI, Anthropic, Google, DeepSeek and Kimi (Moonshot), each with one
query function in `wspmathbench/llm/query.py`.

1. **Register the model.** Add its name to the provider's `'models'` list in
   `BASE_MODEL_REGISTRY` (`wspmathbench/llm/registry.py`). A session refuses a model that is not
   listed.
2. **Describe it to the provider's function.** Each `*_query_async` function in `query.py` has a
   `MODEL_CONFIG` dict with one entry per model: its prices, its limits, and how its reasoning is
   controlled. Add an entry modelled on a neighbouring one. The prices are what `cost_estimate`,
   and so the recorded cost of a session, is computed from. A provider function can also gate a
   request field on a list of model names (for example `EFFORT_MODELS` in `deepseek_query_async`
   decides which models are sent the thinking and effort fields), so read the function for such
   lists and check that the fields you set are actually sent for the new model.
3. **If the model cannot run without thinking,** add it to `NO_THINKING_OFF_MODELS` in
   `registry.py`, so that a configuration that turns thinking off raises at the start.
4. **Run it.** A model without a preset needs `--provider`, and takes its settings from `--effort`
   and `--llm-kwargs` (a JSON object passed to the provider's function):

   ```
   python scripts/run_sessions.py --provider OpenAI --model gpt-5-nano --effort low --tiny-model TinyStories-33M --tier oneshot_spec --smoketest
   ```

   `--smoketest` writes into `results/smoketest/` and adds no row; it still calls the API and
   costs what one call costs. Leave it out for a session that counts.
5. **Optionally add a preset.** An entry in `PRESETS` (`scripts/run_sessions.py`), a pair of the
   provider and the `llm_kwargs`, lets `--model` alone select the settings.

The merged settings of a session are recorded verbatim in the `llm_kwargs` column of its row, and
the analysis names a primary LLM by its model name and the `reasoning_effort` in that column.

### A new provider

1. Write an async function in `query.py` that takes `(prompt, model, **settings)` and returns
   the response contract described at the top of that module: at least `text`,
   `reasoning_text`, the token counts, `cost_estimate`, `status`, `incomplete_reason` and
   `server_provider`. Build its client without automatic retries; the caller retries
   (`registry.retry_llm_call`).
2. Add the provider to `BASE_MODEL_REGISTRY` in `registry.py` with that function and its models.
3. Add the provider to `WSP_LLM_DEFAULTS` in `wspmathbench/benchmark.py`, with the default
   settings merged under the caller's (an empty dict is enough). A session refuses a provider
   that is not there.
4. Add its API key to `.env` and to `.env.example`.

## Adding a tiny model

Everything the benchmark knows about a tiny model is in its entry of
`config.TINY_MODEL_REGISTRY`; the comment above the registry in `wspmathbench/config.py` explains
every field. The steps:

1. **The registry entry.** Copy the entry of the most similar existing model and adjust it:
   - where it comes from: `hf_repo`, `revision`, `model_dir`, and `source_snapshot` and
     `conversion` if `tools/prepare_tiny_models.py` should copy or convert it;
   - how its tensors are named: `config_root`, `editable_prefix`, `layer_stem`,
     `layer_kinds_key`, `layer_kind_labels`, `head_kinds`, `heads_key`,
     `attention_output_suffix`, `residual_writer_suffixes`, `residual_bias_suffixes`,
     `final_norm`, `final_norm_kind`, `post_norms`, `embedding`;
   - which tensors are frozen and how the prompt names them: `frozen`;
   - what it is evaluated on: `tasks` (task to protocol), `default_task`, `default_version`;
   - how it is run: `engine` (`vllm`, the default, or `transformers` for a model vLLM cannot
     load), `max_model_len`, `vllm_kwargs`, `serve_args`, `thinking_mode`;
   - `arch_text`, the one-sentence description of the architecture that the prompt quotes.

   Tensor names and shapes are read from the model's safetensors file at run time, so only the
   names with a special role are written in the entry. The weights must be a single safetensors
   file in a plain folder, in the dtype the model will run in (bfloat16 for all five current
   models).
2. **A protocol.** Add an entry to `config.PROTOCOLS`: the format (`chat` or `completion`), where
   the `\boxed{}` instruction goes, the sampling settings with the new-token limit, and `text`,
   the sentence that describes all of this to the primary LLM. Use the settings the model's
   authors recommend.
3. **A task.** The two existing tasks are `math` (practice on GSM8K and MATH training problems,
   scored on GSM8K-500 and MATH-500) and `basic_arithmetic`. A model that fits one of them needs
   nothing more here. A new task needs an entry in `config.TASKS`, its practice file in
   `data/practice/`, its evaluation set in `data/eval/<name>_Benchmark.jsonl`, and an entry for
   that set in `EVAL_SETS` in both `wspmathbench/llm/benchmarks.py` (how it is asked and graded)
   and `wspmathbench/analysis.py` (how an answer is compared with the reference).
4. **Put the model in place.** Download it at the pinned revision and, if the entry names a
   `source_snapshot`, run `python tools/prepare_tiny_models.py <key>` (`docs/SETUP.md`, step 5).
   A model that needs a conversion other than the two that exist needs a few lines in that tool.
5. **A practice run.** On the Linux side, with the GPU free:

   ```bash
   ~/venvs/vllm/bin/python tools/run_practice_baseline.py --tiny-model <key> --run-name baseline_256
   ```

   Then name that run for the model in `MODEL_PRACTICE_RUNS` in `wspmathbench/benchmark.py`; a
   session looks its practice run up there.
6. **Check it.** `python tests/selftest.py --only-model <key>` runs the architecture-dependent
   checks: the edit rules, the edit language, the prompts for both versions and tiers, and a stub
   session per tier. The self-test also compares each model's rendered prompts with reference
   copies in `tests/reference_prompts/`; a new model has none yet, so expect that check to need a
   reference copy added (read `check_reference_prompts` in `tests/selftest.py` for the file names
   and the placeholder paths the copies are rendered with). Read the rendered prompt of a stub
   smoketest session (`results/smoketest/.../prompt.txt`) in full: it is what a primary LLM will
   see.
7. **Score the unedited model** (`python scripts/score_tiny_model.py --tiny-model <key>`) before
   spending money on sessions, to see that serving and grading work for it.

Two places assume properties of the current models and may need a line for a new architecture:
`dsl._head_layers_text` has wording only for models whose layers all have heads or whose
head-carrying layers are of kind `full_attention`, and `dsl.py`'s handling of the final norm
covers the three kinds named in `final_norm_kind`.

Adding a tiny model changes what a complete evaluation is: `analysis.py` gives a primary LLM a
score only when it has a scored session for every registered tiny model in every tier.
