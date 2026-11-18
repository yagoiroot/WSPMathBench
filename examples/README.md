# Example sessions

Nine of the edit sessions behind the paper's scores, one folder each. A session is one prompt to a
primary LLM and its one reply, which is either an edit specification (`oneshot_spec`) or a Python
program that edits the weights (`oneshot_code`).

| folder | tier | tiny model | primary LLM (effort) | change in score, points |
|---|---|---|---|---|
| `02_...` | code | Qwen3.5-0.8B | deepseek-v4.1-flash-1017 (high) | GSM8K-500 +32.3, MATH-500 +21.2 |
| `12_...` | spec | MobileLLM-R1-950M | deepseek-v4.1-flash-1017 (high) | GSM8K-500 −9.8, MATH-500 −10.6 |
| `23_...` | code | Qwen3.5-0.8B | kimi-k3.1-pro (max) | GSM8K-500 −9.5, MATH-500 −5.8 |
| `28_...` | spec | gemma-3-270m-it | kimi-k3.1-pro (max) | GSM8K-500 +0.4, MATH-500 −1.0 (neither significant) |
| `44_...` | spec | Qwen3.5-0.8B | gpt-6-astra (max) | GSM8K-500 +33.9, MATH-500 +24.7 |
| `48_...` | spec | LFM2.5-350M | gpt-6-astra (max) | GSM8K-500 +1.2, MATH-500 +0.5 (neither significant) |
| `54_...` | spec | MobileLLM-R1-950M | gemini-4-carbon (high) | GSM8K-500 −25.3, MATH-500 −30.6 |
| `72_...` | code | Qwen3.5-0.8B | claude-fable-5-2 (medium) | GSM8K-500 +28.8, MATH-500 +20.7 |
| `76_...` | code | TinyStories-33M | gpt-6-astra-pro (max) | BasicArithmetic +95.5 |

The number at the start of a folder name is the session's ID in the paper's results.

## Files in a folder

| file | what it is |
|---|---|
| `prompt.txt` | the whole prompt the primary LLM received |
| `reply.md` | its whole reply |
| `reasoning.md` | its full reasoning text. Only in folders `02`, `12`, `23` and `28`: DeepSeek and Kimi return their reasoning in full, the other providers return a summary or nothing, so theirs is left out |
| `program_output.txt` | `oneshot_code` only: what the program printed when it ran |
| `session.json` | the session's settings, token counts, cost, the tensors it edited, and the change in the tiny model's score with its standard error |

In `session.json`, `score_change` gives, for each evaluation set, the unedited and edited tiny
model's mean score in percent, the difference in points, the standard error of that difference, and
whether the difference is significant under the paper's test.

## What was changed in these files

Two folder paths of the machine that ran the sessions appear in the `oneshot_code` prompts (the
prompt tells the program where the model's files and the practice problems are) and in the replies
that quote them. They are replaced by `<MODELS_DIR>` and `<BENCHMARK_DIR>`. Nothing else was
altered.
