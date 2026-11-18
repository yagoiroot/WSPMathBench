"""Every path, limit, version string, task, protocol and the tiny-model registry of WSPMathBench.

Where it sits in the flow: imported first by every other module.

Machine-specific settings (where the tiny models are, which Python runs vLLM, ...) are read from
environment variables, each with a default. They can also be written in a `.env` file at the
repository root, which this module reads (see `.env.example`); a variable already set in the
environment wins over the file. API keys in that file reach the provider clients the same way.

Reads `.env`, writes nothing. Standard library only and no relative imports, because the tools that
run on the Linux side load this file by its path.
"""
import json
import os
import re

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)


def _load_env_file(path):
    """KEY=VALUE lines of a .env file into os.environ, without replacing a variable already set."""
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if key.startswith('export '):
                key = key[len('export '):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            if value:  # an empty value means "not set", so a missing key is reported as missing
                os.environ.setdefault(key, value)


_load_env_file(os.path.join(REPO_ROOT, '.env'))


def _setting(name, default):
    """An environment variable, or its default when it is unset or empty."""
    return os.environ.get(name) or default


# --- Versions. A prompt version names the SCOPE of the prompt: v1 is written around repairing a
# fault, v2 is open-ended. The text that depends on the tiny model (architecture, edit-language
# details, task sentences) varies under a label appended to the version, so a row's prompt_version
# is for example 'v2-gemma-3-270m-it' (version_label below). A changed template means a new version
# and a new file, never an in-place change of what an existing row's primary LLM saw.
BENCHMARK_VERSION = 'v1'
PROMPT_VERSIONS = ('v1', 'v2')
DSL_VERSION = 'v1'

TIERS = ('oneshot_spec', 'oneshot_code')
EXPECTED_BLOCK_KIND = {'oneshot_spec': 'json', 'oneshot_code': 'python'}

# --- Paths inside the repository
PROMPTS_DIR = os.path.join(REPO_ROOT, 'prompts')
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
EVAL_DIR = os.path.join(DATA_DIR, 'eval')                    # the held-out evaluation sets
PRACTICE_DIR = os.path.join(DATA_DIR, 'practice')            # the practice pools
PRACTICE_RUNS_DIR = os.path.join(DATA_DIR, 'practice_runs')  # the unedited models' practice answers
KERNEL_RUNNER_FILE = os.path.join(PACKAGE_DIR, 'kernel_runner.py')

# --- Where results are written (WSPMB_RESULTS_DIR; by default <repository>/results)
RESULTS_DIR = _setting('WSPMB_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))
SESSIONS_DIR = os.path.join(RESULTS_DIR, 'sessions')
SMOKETEST_DIR = os.path.join(RESULTS_DIR, 'smoketest')
MASTER_CSV = os.path.join(RESULTS_DIR, 'Master_WSP_Edits.csv')  # one row per edit session
# The scores of the tiny models: Master_LLM_<evaluation set>_Results.csv, one row per scoring pass,
# and <evaluation set>_model_answers/<row ID>_<evaluation set>_model_answers.txt with the full answers.
SCORES_DIR = os.path.join(RESULTS_DIR, 'scores')
COMPILED_METRICS_FILE = os.path.join(RESULTS_DIR, 'compiled_wsp_benchmark_metrics.parquet')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
ERROR_LOG_DIR = os.path.join(RESULTS_DIR, 'error_logs')

# --- The tiny models and the edited checkpoints (WSPMB_MODELS_DIR). On Windows they are read by
# vLLM inside WSL through /mnt/<drive>/, so they sit on a local disk as plain files. About 2 GB per
# built checkpoint.
MODELS_DIR = _setting('WSPMB_MODELS_DIR', os.path.join(os.path.expanduser('~'), 'wspmathbench_models'))
HF_CACHE_DIR = _setting('WSPMB_HF_CACHE_DIR',
                        os.environ.get('HF_HUB_CACHE')
                        or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'))
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'wsp_checkpoints')
SMOKETEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, 'smoketest')

# --- The Linux side. vLLM does not run on Windows, so on Windows the tiny models are served, and
# the oneshot_code programs are run, inside WSL2; on Linux the same commands run directly in bash.
# These are Linux paths, expanded by bash (a leading ~ is the Linux user's home).
IS_WINDOWS = os.name == 'nt'
WSL_EXE = 'wsl.exe'
VLLM_VENV = _setting('WSPMB_VLLM_VENV', '~/venvs/vllm')                  # has vllm, torch, transformers
VLLM_PYTHON = f'{VLLM_VENV}/bin/python'
VLLM_ACTIVATE = f'{VLLM_VENV}/bin/activate'
SANDBOX_PYTHON = _setting('WSPMB_SANDBOX_PYTHON', '~/venvs/wsp_sandbox/bin/python')  # see docs/SETUP.md
SERVER_LOG_DIR = _setting('WSPMB_SERVER_LOG_DIR', '~/wspmathbench_server_logs')
SERVER_PORT = int(_setting('WSPMB_SERVER_PORT', '8000'))
# Environment variables set in front of `vllm serve`. Pinned memory is off by default under WSL2.
SERVER_ENV = 'VLLM_WSL2_ENABLE_PIN_MEMORY=1 ' if IS_WINDOWS else ''

# --- The layout of a session folder
SESSION_PROMPT_FILE = 'prompt.txt'
SESSION_TURNS_DIR = 'turns'
SESSION_TOOL_CALLS_FILE = 'tool_calls.jsonl'
SESSION_WORK_DIR = 'work'
SESSION_EDITED_DIR = 'edited'
SESSION_MANIFEST_FILE = os.path.join('edit', 'manifest.json')
SESSION_STATE_FILE = 'state.json'
TURN_ANSWER_FILE = 'answer.md'
TURN_REASONING_FILE = 'reasoning.md'
TURN_RUN_FILE = 'run.json'

# --- Files of a model directory, and of a built checkpoint
MODEL_CONFIG_FILE = 'config.json'
MODEL_GENERATION_CONFIG_FILE = 'generation_config.json'
MODEL_TOKENIZER_FILE = 'tokenizer.json'
CHECKPOINT_WEIGHTS_FILE = 'model.safetensors'
CHECKPOINT_RECORD_FILE = 'edits.json'
# The base's index names its shard file. Copied next to a differently named model.safetensors it
# would point loaders at a file that is not there, and a single file needs no index anyway.
CHECKPOINT_NOT_COPIED = ('model.safetensors.index.json',)
# A checkpoint gets exactly one weights file, the new model.safetensors. Any other weights file
# in the base folder is left behind.
WEIGHTS_EXTENSIONS = ('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.h5', '.msgpack', '.onnx')

# --- Practice runs
BASELINE_FILE = 'baseline.jsonl'
SUMMARY_FILE = 'summary.json'
# A run under this prefix holds evaluation-set problems (tools/make_dev_fixture.py), so it may
# only ever be used for a smoketest.
DEV_FIXTURE_PREFIX = 'DEVFIXTURE'

# --- Diagnostic pack
DIAG_DEFAULTS = {'k': 8, 'head_chars': 600, 'tail_chars': 400, 'seed': 0}

# --- Edit language limits
MAX_OPS = 8
MAX_INDICES = 64
MAX_FACTOR = 4.0  # upper bound of both `factor` and `strength`; the lower bound is 0

BOXED_INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'

# --- Tasks. A task is a practice pool, a held-out evaluation, a grader and the task sentences of
# the prompt. Each *_text value is quoted in the prompt as written.
#   practice_sources  source label -> practice file, in the order a baseline run samples them
#   grader      'boxed' (last \boxed{} answer, math-verify) or 'first_number'
#   style             how the diagnostic pack shows an answer: 'worked' (long, with a reference
#                     solution) or 'short' (a few tokens, shown whole)
#   eval_sets         the evaluation sets (data/eval/<name>_Benchmark.jsonl) that score it
TASKS = {
    'math': {
        'practice_sources': {'gsm8k': 'gsm8k_train.jsonl', 'math': 'math_train.jsonl'},
        'grader': 'boxed',
        'style': 'worked',
        'eval_sets': ('GSM8K_500', 'MATH500'),
        'goal_text': 'mathematics',
        'evaluation_text': 'MATH-500 and GSM8K',
        'unseen_text': 'You never see these benchmarks.',
        'practice_text': ('7,469 grade-school problems (GSM8K train) and 7,275 competition problems (MATH '
                          'train, with subject and level 1-5), each with a reference answer and a worked '
                          'reference solution.'),
        'holdout_text': 'None of them is in the evaluation.',
        'example_fields_text': ("the problem, the reference answer and the reference solution, the model's "
                                "answer (first {head_chars} and last {tail_chars} characters) and the outcome"),
        'practice_files_text': ('gsm8k_train.jsonl and math_train.jsonl, one JSON object per line with id, '
                                'source, problem, answer and solution'),
    },
    # The practice pool and the evaluation set overlap, because problems this easy are few.
    # holdout_text says so instead of claiming a held-out set.
    'basic_arithmetic': {
        'practice_sources': {'symbolic': 'basic_arithmetic_practice.jsonl', 'word': 'basic_arithmetic_practice.jsonl'},
        'grader': 'first_number',
        'style': 'short',
        'eval_sets': ('BasicArithmetic',),
        'goal_text': 'basic arithmetic',
        'evaluation_text': ('BasicArithmetic (500 very easy questions: 250 additions, subtractions, '
                            'multiplications and divisions, mostly of single-digit numbers, each written '
                            'once with digits and symbols and once in words; the symbolic questions '
                            'come with and without spaces, "2 + 2 =" and "2+2=", and multiplication is '
                            'written "x" or "*")'),
        'unseen_text': 'You are not told which questions this benchmark holds.',
        'practice_text': ('1,000 very easy arithmetic questions: 500 problems (every addition and '
                          'multiplication of two single-digit numbers, every subtraction of two single-digit '
                          'numbers with a non-negative result, every division of two single-digit numbers '
                          'that comes out whole, and 213 additions, subtractions and divisions with a '
                          'two-digit number), each written once with digits and symbols (half with '
                          'spaces, "2 + 2 =", half without, "2+2="; multiplication as "x" or "*", '
                          'division as "/") and once in words ("two plus two equals"), with its '
                          'reference answer.'),
        'holdout_text': 'Problems this easy are few, so many of them are also in the evaluation.',
        'example_fields_text': ("the question exactly as the model was given it, the reference answer, the "
                                "model's whole continuation and the outcome"),
        'practice_files_text': ('basic_arithmetic_practice.jsonl, one JSON object per line with id, source '
                                '(symbolic or word), problem, answer, operation, a, b and, for a symbolic '
                                'question, symbol and spacing (spaced or tight)'),
    },
}

# --- Evaluation protocols. Each model runs with its author's recommended settings; Qwen keeps
# the think35 protocol of Takehi et al. (Qwen3.5 with thinking). One protocol per tiny model, named
# after it. The new-token limit is as high as the model's context allows: 30,000 of 32,768, the
# rest being room for the problem. TinyStories needs 16. The prompt quotes `text`; the practice
# baseline tool and the scoring code run the rest, so the answers shown to the primary LLM were
# produced the way the edited model will be scored.
#   format            'chat' (the model's chat template) or 'completion' (the text as given)
#   instruction_role  where BOXED_INSTRUCTION goes: 'user' (after the problem and a newline),
#                     'system' (the system message) or None
#   template_kwargs   extra arguments of the chat template
#   sampling          vLLM SamplingParams; max_tokens is the protocol's new-token limit
PROTOCOLS = {
    'think35': {
        'text': ('the chat template with thinking enabled; the user message is the problem, a '
                 'newline and "Please reason step by step, and put your final answer within '
                 '\\boxed{}."; temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at '
                 'most 32,768 new tokens for reasoning and answer together; the last \\boxed{} '
                 'answer is graded against the reference answer.'),
        'format': 'chat', 'instruction_role': 'user', 'template_kwargs': {'enable_thinking': True},
        'sampling': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                     'presence_penalty': 1.5, 'max_tokens': 32768},
    },
    # Meta's own evaluation script for MobileLLM-R1 (evaluation/run_eval.sh in its GitHub repository)
    # samples at temperature 0.6, top_p 0.95. Its chat template inserts this same
    # instruction as the default system message.
    'mobilellm_r1_950m': {
        'text': ('the chat template; the system message is "Please reason step by step, and put your '
                 'final answer within \\boxed{}." and the user message is the problem; temperature 0.6, '
                 'top_p 0.95, at most 30,000 new tokens; the last \\boxed{} answer is graded against the '
                 'reference answer.'),
        'format': 'chat', 'instruction_role': 'system', 'template_kwargs': {},
        'sampling': {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 30000},
    },
    # LFM2.5's card: temperature 0.1, top_k 50, repetition_penalty 1.05.
    'lfm2.5_350m': {
        'text': ('the chat template; the user message is the problem, a newline and "Please reason step '
                 'by step, and put your final answer within \\boxed{}."; temperature 0.1, top_k 50, '
                 'repetition penalty 1.05, at most 30,000 new tokens; the last \\boxed{} answer is graded '
                 'against the reference answer.'),
        'format': 'chat', 'instruction_role': 'user', 'template_kwargs': {},
        'sampling': {'temperature': 0.1, 'top_k': 50, 'repetition_penalty': 1.05, 'max_tokens': 30000},
    },
    # Gemma 3 270M-it's generation_config: sampling with top_k 64, top_p 0.95 (temperature 1.0 by
    # default).
    'gemma3_270m_it': {
        'text': ('the chat template; the user message is the problem, a newline and "Please reason step '
                 'by step, and put your final answer within \\boxed{}."; temperature 1.0, top_k 64, '
                 'top_p 0.95, at most 30,000 new tokens; the last \\boxed{} answer is graded against the '
                 'reference answer.'),
        'format': 'chat', 'instruction_role': 'user', 'template_kwargs': {},
        'sampling': {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95, 'max_tokens': 30000},
    },
    # TinyStories-33M is not a chat model. Temperature 0.1 only so that passes are not identical;
    # 16 new tokens hold everything an answer needs.
    'tinystories_33m': {
        'text': ('the model continues the question text exactly as written (no chat template, nothing '
                 'added before or after it); temperature 0.1, at most 16 new tokens; the first number in '
                 'the continuation, written in digits or in words, is graded against the reference answer.'),
        'format': 'completion', 'instruction_role': None, 'template_kwargs': {},
        'sampling': {'temperature': 0.1, 'max_tokens': 16},
    },
}

# --- Running the tiny model in vLLM. Every tiny model runs in bfloat16.
BASELINE_VLLM = {'dtype': 'bfloat16', 'gpu_memory_utilization': 0.85}
BASELINE_ENV = {'VLLM_WSL2_ENABLE_PIN_MEMORY': '1'}


def _snapshot(repo, revision):
    return os.path.join(HF_CACHE_DIR, 'models--' + repo.replace('/', '--'), 'snapshots', revision)


# --- The tiny models. Tensor names and shapes are read from
# the safetensors header at run time; only the names with a special role are listed here.
#   label               the model part of a version label (version_label)
#   model_dir           plain files; tools/prepare_tiny_models.py builds it from source_snapshot
#   conversion          what prepare_tiny_models.py did to the published weights, if anything.
#                       The benchmark's base is then a derived file, recorded in
#                       <model_dir>/wsp_source.json with the source file's SHA-256.
#   config_root         the key of config.json holding the language model's config, or None
#   editable_prefix     tensors under it are the language model; all but `frozen` are editable
#   layer_stem          the name of layer N's tensors up to the layer index
#   layer_kinds_key     the config field listing each layer's kind; layer_kind_labels is how the
#                       prompt's tensor table names each kind
#   head_kinds          layer kinds whose attention output is split into heads (zero_head)
#   heads_key           the config field with the number of heads
#   attention_output_suffix, residual_writer_suffixes, residual_bias_suffixes
#                       the tensors zero_head and project_out accept
#   final_norm_kind     how fold_final_norm uses the final norm: 'rms' (weight), 'rms_one_plus'
#                       (1 + weight) or 'layernorm' (weight; centring and bias ignored)
#   post_norms          for a model with a norm between a residual writer and the residual
#                       stream: writer suffix -> that norm's suffix (gain 1 + weight)
#   frozen              name -> how the prompt refers to it. lm_head is absent where the embedding
#                       is tied but stays listed, so an edit naming it is refused as frozen.
#   tasks               task -> protocol; default_task and default_version are what a session
#                       uses unless told otherwise
#   engine              what runs the model: 'vllm' (the default when the key is absent) or
#                       'transformers'
#   max_model_len, vllm_kwargs, serve_args
#                       how the engine loads it: offline (the baseline tool) and as a server
#                       (llm/query.py, wsp_local_model)
TINY_MODEL_REGISTRY = {
    'Qwen3.5-0.8B': {
        'label': 'Qwen3.5-0.8B',
        'hf_repo': 'Qwen/Qwen3.5-0.8B',
        'revision': '2fc06364715b967f1860aea9cf38778875588b17',
        'model_dir': os.path.join(MODELS_DIR, 'Qwen3.5-0.8B'),
        'source_snapshot': None, 'conversion': None,
        'config_root': 'text_config',
        'editable_prefix': 'model.language_model.',
        'layer_stem': 'model.language_model.layers.{layer}',
        'layer_kinds_key': 'layer_types',
        'layer_kind_labels': {'full_attention': 'full attention', 'linear_attention': 'linear attention'},
        'head_kinds': ('full_attention',),
        'heads_key': 'num_attention_heads',
        'attention_output_suffix': '.self_attn.o_proj.weight',
        'residual_writer_suffixes': ('.mlp.down_proj.weight', '.self_attn.o_proj.weight'),
        'residual_bias_suffixes': (),
        'final_norm': 'model.language_model.norm.weight',
        'final_norm_kind': 'rms',
        'post_norms': {},
        'embedding': 'model.language_model.embed_tokens.weight',
        'frozen': {
            'model.language_model.norm.weight': 'the final norm',
            'model.language_model.embed_tokens.weight': 'the input embedding',
            'lm_head.weight': 'the output layer',
        },
        'tasks': {'math': 'think35'},
        'default_task': 'math',
        'default_version': 'v1',
        'max_model_len': 34816,
        'vllm_kwargs': {},
        # 128 concurrent sequences, text-only load and Triton attention: the fastest settings
        # found that leave the outputs unchanged, with room for the protocol's 32,768 new tokens.
        # The recurrent state stays float32.
        'serve_args': ('--max-model-len 34816 --max-num-seqs 128 --reasoning-parser qwen3 '
                       '--language-model-only --attention-backend TRITON_ATTN '
                       '--compilation-config \'{"max_cudagraph_capture_size": 128}\''),
        'thinking_mode': 'enable_thinking',
        'arch_text': ('24 layers of width 1,024; layers 3, 7, 11, 15, 19 and 23 are full-attention '
                      'layers with 8 query heads and 2 key/value heads of size 256 and a gated '
                      'output, the other 18 are Gated DeltaNet linear-attention layers with 16 '
                      'heads of size 128; each layer has a SwiGLU MLP of width 3,584; RMSNorm; '
                      'vocabulary 248,320; input embedding and output layer tied'),
    },
    'MobileLLM-R1-950M': {
        'label': 'MobileLLM-R1-950M',
        'hf_repo': 'facebook/MobileLLM-R1-950M',
        'revision': 'aa7d61df54d764738d32c220e107ed5f292d56eb',
        'model_dir': os.path.join(MODELS_DIR, 'MobileLLM-R1-950M-bf16'),
        'source_snapshot': _snapshot('facebook/MobileLLM-R1-950M', 'aa7d61df54d764738d32c220e107ed5f292d56eb'),
        'conversion': 'float32 safetensors cast to bfloat16',
        'config_root': None,
        'editable_prefix': 'model.',
        'layer_stem': 'model.layers.{layer}',
        'layer_kinds_key': 'layer_types',
        'layer_kind_labels': {'full_attention': 'full attention'},
        'head_kinds': ('full_attention',),
        'heads_key': 'num_attention_heads',
        'attention_output_suffix': '.self_attn.o_proj.weight',
        'residual_writer_suffixes': ('.feed_forward.down_proj.weight', '.self_attn.o_proj.weight'),
        'residual_bias_suffixes': (),
        'final_norm': 'model.norm.weight',
        'final_norm_kind': 'rms',
        'post_norms': {},
        'embedding': 'model.embed_tokens.weight',
        'frozen': {
            'model.norm.weight': 'the final norm',
            'model.embed_tokens.weight': 'the input embedding',
            'lm_head.weight': 'the output layer',
        },
        'tasks': {'math': 'mobilellm_r1_950m'},
        'default_task': 'math',
        'default_version': 'v2',
        'max_model_len': 32768,
        'vllm_kwargs': {},
        'serve_args': '--dtype bfloat16 --max-model-len 32768 --max-num-seqs 128',
        'thinking_mode': 'none',
        'arch_text': ('22 full-attention layers of width 1,536 with 24 query heads and 6 key/value '
                      'heads of size 64 and normalised queries and keys; each layer has a SwiGLU MLP '
                      'of width 6,144; RMSNorm; vocabulary 128,256; input embedding and output layer '
                      'tied'),
    },
    'LFM2.5-350M': {
        'label': 'LFM2.5-350M',
        'hf_repo': 'LiquidAI/LFM2.5-350M',
        'revision': '9921cd4d82c0e3986887aa4b376f9446729ef8a3',
        'model_dir': os.path.join(MODELS_DIR, 'LFM2.5-350M'),
        'source_snapshot': _snapshot('LiquidAI/LFM2.5-350M', '9921cd4d82c0e3986887aa4b376f9446729ef8a3'),
        'conversion': None,
        'config_root': None,
        'editable_prefix': 'model.',
        'layer_stem': 'model.layers.{layer}',
        'layer_kinds_key': 'layer_types',
        'layer_kind_labels': {'full_attention': 'full attention', 'conv': 'gated convolution'},
        'head_kinds': ('full_attention',),
        'heads_key': 'num_attention_heads',
        'attention_output_suffix': '.self_attn.out_proj.weight',
        'residual_writer_suffixes': ('.feed_forward.w2.weight', '.self_attn.out_proj.weight',
                                     '.conv.out_proj.weight'),
        'residual_bias_suffixes': (),
        'final_norm': 'model.embedding_norm.weight',  # applied after the last layer, despite the name
        'final_norm_kind': 'rms',
        'post_norms': {},
        'embedding': 'model.embed_tokens.weight',
        'frozen': {
            'model.embedding_norm.weight': 'the final norm',
            'model.embed_tokens.weight': 'the input embedding',
            'lm_head.weight': 'the output layer',
        },
        'tasks': {'math': 'lfm2.5_350m'},
        'default_task': 'math',
        'default_version': 'v2',
        'max_model_len': 32768,
        'vllm_kwargs': {},
        'serve_args': '--dtype bfloat16 --max-model-len 32768 --max-num-seqs 128',
        'thinking_mode': 'none',
        'arch_text': ('16 layers of width 1,024; layers 2, 5, 8, 10, 12 and 14 are full-attention '
                      'layers with 16 query heads and 8 key/value heads of size 64 and normalised '
                      'queries and keys, the other 10 are gated convolution layers with kernel size 3; '
                      'each layer has a SwiGLU MLP of width 4,608; RMSNorm; vocabulary 65,536; input '
                      'embedding and output layer tied'),
    },
    'gemma-3-270m-it': {
        'label': 'gemma-3-270m-it',
        'hf_repo': 'google/gemma-3-270m-it',
        'revision': 'ac82b4e820549b854eebf28ce6dedaf9fdfa17b3',
        'model_dir': os.path.join(MODELS_DIR, 'gemma-3-270m-it'),
        'source_snapshot': _snapshot('google/gemma-3-270m-it', 'ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'),
        'conversion': None,
        'config_root': None,
        'editable_prefix': 'model.',
        'layer_stem': 'model.layers.{layer}',
        'layer_kinds_key': 'layer_types',
        'layer_kind_labels': {'full_attention': 'attention over the whole context',
                              'sliding_attention': 'attention over a sliding window'},
        'head_kinds': ('full_attention', 'sliding_attention'),
        'heads_key': 'num_attention_heads',
        'attention_output_suffix': '.self_attn.o_proj.weight',
        'residual_writer_suffixes': ('.mlp.down_proj.weight', '.self_attn.o_proj.weight'),
        'residual_bias_suffixes': (),
        'final_norm': 'model.norm.weight',
        'final_norm_kind': 'rms_one_plus',
        # Each block's output passes through an RMSNorm with gain (1 + weight) before it is added
        # to the residual stream.
        'post_norms': {'.mlp.down_proj.weight': '.post_feedforward_layernorm.weight',
                       '.self_attn.o_proj.weight': '.post_attention_layernorm.weight'},
        'embedding': 'model.embed_tokens.weight',
        'frozen': {
            'model.norm.weight': 'the final norm',
            'model.embed_tokens.weight': 'the input embedding',
            'lm_head.weight': 'the output layer',
        },
        'tasks': {'math': 'gemma3_270m_it'},
        'default_task': 'math',
        'default_version': 'v2',
        'max_model_len': 32768,
        'vllm_kwargs': {},
        'serve_args': '--dtype bfloat16 --max-model-len 32768 --max-num-seqs 128',
        'thinking_mode': 'none',
        'arch_text': ('18 attention layers of width 640 with 4 query heads and 1 key/value head of '
                      'size 256 and normalised queries and keys; layers 5, 11 and 17 attend to the '
                      'whole context, the other 15 to a sliding window of 512 tokens; each layer has a '
                      'gated GELU MLP of width 2,048; RMSNorm with gain (1 + weight) before and after '
                      'each attention and MLP block; vocabulary 262,144; input embedding and output '
                      'layer tied'),
    },
    'TinyStories-33M': {
        'label': 'TinyStories-33M',
        'hf_repo': 'roneneldan/TinyStories-33M',
        'revision': '2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8',
        'model_dir': os.path.join(MODELS_DIR, 'TinyStories-33M-bf16'),
        'source_snapshot': _snapshot('roneneldan/TinyStories-33M', '2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8'),
        'conversion': 'float32 pytorch_model.bin cast to bfloat16 safetensors, non-weight buffers dropped',
        'config_root': None,
        'editable_prefix': 'transformer.',
        'layer_stem': 'transformer.h.{layer}',
        'layer_kinds_key': 'attention_layers',
        'layer_kind_labels': {'global': 'attention over the whole context',
                              'local': 'attention over a window'},
        'head_kinds': ('global', 'local'),
        'heads_key': 'num_heads',
        'attention_output_suffix': '.attn.attention.out_proj.weight',
        'residual_writer_suffixes': ('.mlp.c_proj.weight', '.attn.attention.out_proj.weight'),
        'residual_bias_suffixes': ('.mlp.c_proj.bias', '.attn.attention.out_proj.bias'),
        'final_norm': 'transformer.ln_f.weight',
        'final_norm_kind': 'layernorm',
        'post_norms': {},
        'embedding': 'transformer.wte.weight',
        # The position embedding is an input table like the token embedding.
        'frozen': {
            'transformer.ln_f.weight': 'the final norm',
            'transformer.ln_f.bias': "the final norm's bias",
            'transformer.wte.weight': 'the input embedding',
            'transformer.wpe.weight': 'the position embedding',
            'lm_head.weight': 'the output layer',
        },
        'tasks': {'basic_arithmetic': 'tinystories_33m'},
        'default_task': 'basic_arithmetic',
        'default_version': 'v2',
        # GPT-Neo is not in vLLM 0.29.0's registry and its transformers implementation is refused
        # by vLLM's generic backend, so this model runs in plain
        # transformers: tools/transformers_engine.py, offline and behind tools/serve_transformers.py.
        'engine': 'transformers',
        'max_model_len': 2048,
        'vllm_kwargs': {},
        'serve_args': '--dtype bfloat16',
        'thinking_mode': 'none',
        'arch_text': ('4 layers of width 768 with 16 heads of size 48; layers 0 and 2 attend to the '
                      'whole context, layers 1 and 3 to a window of 256 tokens; each layer has a GELU '
                      'MLP of width 3,072; LayerNorm with bias; learned position embeddings for 2,048 '
                      'positions; vocabulary 50,257; input embedding and output layer tied'),
    },
}

# --- The oneshot_code kernel. timeout_s and mem_bytes are also quoted in the prompt, so changing
# either changes what the primary LLM is told. The paper's sessions ran with 900 s and 56 GiB (the
# machine had 64 GB); WSPMB_SANDBOX_MEM_GB and WSPMB_SANDBOX_THREADS adapt them to another machine.
SANDBOX = {
    'timeout_s': 900,
    'mem_bytes': int(_setting('WSPMB_SANDBOX_MEM_GB', '56')) << 30,
    'max_output_chars': 10_000,
    'threads': int(_setting('WSPMB_SANDBOX_THREADS', '8')),
    'python': SANDBOX_PYTHON,
    'mode': 'subprocess',
    # The runner stops the program itself at timeout_s. If the program is stuck in C code, Linux
    # `timeout` kills it at timeout_s + hard_kill_margin_s, because killing wsl.exe from Windows
    # does not reliably kill the Linux process. The caller gives up at timeout_s + wsl_margin_s.
    'hard_kill_margin_s': 20,
    'wsl_margin_s': 30,
}


def shell_command(bash_command):
    """The argument list that runs a bash command line on the Linux side: through wsl.exe on
    Windows, directly in bash elsewhere."""
    if IS_WINDOWS:
        return [WSL_EXE, '-e', 'bash', '-c', bash_command]
    return ['bash', '-c', bash_command]


def tiny_entry(tiny_model):
    """The registry entry of a tiny model, or a ValueError naming the registered ones."""
    if tiny_model not in TINY_MODEL_REGISTRY:
        raise ValueError(f'`tiny_model` must be one of {list(TINY_MODEL_REGISTRY)}, got {tiny_model!r}.')
    return TINY_MODEL_REGISTRY[tiny_model]


def resolve_task(tiny_model, task=None):
    """(task, protocol name) for a tiny model; task=None means the model's default task."""
    entry = tiny_entry(tiny_model)
    task = entry['default_task'] if task is None else task
    if task not in entry['tasks']:
        raise ValueError(f"`task` must be one of {list(entry['tasks'])} for {tiny_model}, got {task!r}.")
    return task, entry['tasks'][task]


def resolve_version(tiny_model, prompt_version=None):
    """The prompt version (scope) of a session; None means the model's default."""
    version = tiny_entry(tiny_model)['default_version'] if prompt_version is None else prompt_version
    if version not in PROMPT_VERSIONS:
        raise ValueError(f'`prompt_version` must be one of {PROMPT_VERSIONS}, got {version!r}.')
    return version


def version_label(version, tiny_model):
    """What a row records as prompt_version or dsl_version: the version and the model's label."""
    return f"{version}-{tiny_entry(tiny_model)['label']}"


def practice_files(task):
    """The practice files of a task, each once, in sampling order."""
    return tuple(dict.fromkeys(TASKS[task]['practice_sources'].values()))


def practice_run_dir(tiny_model, protocol, practice_run):
    return os.path.join(PRACTICE_RUNS_DIR, tiny_model, protocol, practice_run)


def to_wsl_path(path):
    """A path as the Linux side sees it: the /mnt/<drive>/... form of a Windows path. A path that
    is already POSIX is unchanged, so on Linux this returns the path itself."""
    path = str(path)
    if path.startswith('/'):
        return path
    # A path that already has a drive letter is used as written: inside WSL (where the tools also
    # load this file) os.path.abspath would prefix it with the Linux working directory.
    drive_path = r'^([A-Za-z]):[\\/]*(.*)$'
    match = re.match(drive_path, path) or re.match(drive_path, os.path.abspath(path))
    if match is None:
        raise ValueError(f'Cannot translate {path!r} to a WSL path: it has no drive letter.')
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def stop_token_ids(model_dir, entry):
    """Every end-of-sequence id the model's files name: generation_config.json's, then config.json's
    (Qwen's release has no generation_config.json, so vLLM would stop only on the tokenizer's
    <|im_end|>; the export of Takehi et al. also stopped on config.json's <|endoftext|>). entry: the model's
    registry entry, or None to read config.json at its top level."""
    ids = []
    for file_name, root in ((MODEL_GENERATION_CONFIG_FILE, None),
                            (MODEL_CONFIG_FILE, entry['config_root'] if entry else None)):
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            config = json.load(f)
        value = (config[root] if root else config).get('eos_token_id')
        ids += value if isinstance(value, list) else [] if value is None else [value]
    return list(dict.fromkeys(ids))


def checkpoint_dir(checkpoint):
    """The folder of a checkpoint: a session ID (its folder under CHECKPOINT_DIR) or any model folder."""
    return os.path.join(CHECKPOINT_DIR, str(checkpoint)) if str(checkpoint).isdigit() else str(checkpoint)


def read_checkpoint_record(model_dir):
    """A built checkpoint's edits.json (session ID, tiny model, task, edited tensors), or None for
    a folder that has none, such as an unedited model's."""
    record_path = os.path.join(model_dir, CHECKPOINT_RECORD_FILE)
    if not os.path.exists(record_path):
        return None
    with open(record_path, encoding='utf-8') as f:
        return json.load(f)
