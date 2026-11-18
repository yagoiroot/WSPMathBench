"""Render a prompt from a versioned template plus the pieces other modules supply.

Where it sits in the flow: step 3 of the session. The whole text a model sees can be read in one
prompts/<tier>_<version>.txt file plus these pieces, each of which comes from its owner:
    the architecture      cfg.TINY_MODEL_REGISTRY (arch_text, the frozen tensors)
    the task sentences    cfg.TASKS (what to improve, the evaluation, the practice pool)
    the protocol          cfg.PROTOCOLS
    the tensor table      rendered here, from the TinyModel (never typed in)
    the practice summary  diagnostics
    the examples          diagnostics
    the edit language     dsl.render_dsl_text
    the prelude           kernel_runner.PRELUDE_DOC

A version is the scope of the prompt (v1 repair-oriented, v2 open-ended): one template per version
and tier, filled per tiny model and task. The prompt is part of the treatment: changing what a
version says means a new version and a new file, never a change to what an existing row's primary
LLM saw.

Reads prompts/*.txt. Writes nothing (the session saves the rendered prompt.txt).
"""
import os

from wspmathbench import config as cfg
from wspmathbench import dsl
from wspmathbench import kernel_runner

# Appended to the original prompt when a reply cannot be used. The query wrappers are single-turn,
# so a retry is a fresh call with this longer prompt. The heading is stub.RETRY_MARKER.
RETRY_SUFFIX = """

PREVIOUS ATTEMPT (not applied)
{reply}

ERRORS
{errors}

Reply again with exactly one block."""

# How the prompt names what turns a problem into the model's input, by protocol format.
_PROMPT_FORMAT_TEXT = {'chat': 'chat template', 'completion': 'prompt format'}


def render(tier, tiny, pack, paths, version):
    """The first prompt of a session.

    pack: from diagnostics.build; it names the task and the protocol.
    paths: {'model_dir', 'practice_dir', 'work_dir'}, as the program would see them (WSL paths).
    version: a member of cfg.PROMPT_VERSIONS.
    A placeholder without a value raises KeyError, so a template can never go out half filled.
    """
    entry = tiny.entry
    task = cfg.TASKS[pack['task']]
    protocol = cfg.PROTOCOLS[pack['protocol']]
    limits = cfg.SANDBOX
    values = {
        'tiny_name': tiny.key,
        'arch_text': entry['arch_text'],
        'goal_text': task['goal_text'],
        'evaluation_text': task['evaluation_text'],
        'protocol_text': protocol['text'],
        'prompt_format_text': _PROMPT_FORMAT_TEXT[protocol['format']],
        'unseen_text': task['unseen_text'],
        'frozen_text': render_frozen_text(entry['frozen']),
        'width': tiny.width,
        'tensor_table': render_tensor_table(tiny),
        'practice_text': task['practice_text'],
        'holdout_text': task['holdout_text'],
        'n_run': pack['n_run'],
        'practice_summary': pack['summary_text'],
        'k': pack['k'],
        'example_fields_text': task['example_fields_text'].format(head_chars=pack['head_chars'],
                                                                  tail_chars=pack['tail_chars']),
        'examples': pack['examples_text'],
        'max_ops': cfg.MAX_OPS,
        'dsl_text': dsl.render_dsl_text(tiny),
        'timeout_s': limits['timeout_s'],
        'mem_gb': limits['mem_bytes'] >> 30,
        'practice_files_text': task['practice_files_text'],
        'prelude_doc': kernel_runner.PRELUDE_DOC,
        **paths,
    }
    return _read_template(f'{tier}_{version}.txt').format_map(values)


def system_prompt(version):
    return _read_template(f'system_{version}.txt')


def retry_prompt(prompt, reply, errors):
    """The original prompt plus RETRY_SUFFIX. errors is a list of sentences."""
    return prompt + RETRY_SUFFIX.format_map({'reply': reply, 'errors': '\n'.join(errors)})


def render_tensor_table(tiny):
    """One line per editable tensor, grouped under a header line per layer, then any others."""
    lines, current_layer = [], None
    for name in tiny.names:
        layer = tiny.layer_of(name)
        if layer != current_layer:
            lines.append('other tensors' if layer is None else f'layer {layer} ({tiny.layer_label(layer)})')
            current_layer = layer
        lines.append(f'  {name} {tiny.shapes[name]}')
    return '\n'.join(lines)


def render_frozen_text(frozen):
    """'the final norm (name), the input embedding (name) and ...' from the registry's frozen list."""
    parts = [f'{label} ({name})' for name, label in frozen.items()]
    return ', '.join(parts[:-1]) + ' and ' + parts[-1]


def _read_template(file_name):
    with open(os.path.join(cfg.PROMPTS_DIR, file_name), encoding='utf-8') as f:
        return f.read().strip()
