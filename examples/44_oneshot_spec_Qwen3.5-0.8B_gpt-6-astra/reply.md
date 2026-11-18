Nearly 90% of responses hit the token limit, and Example 5 already obtains the correct answer before repeatedly emitting “Wait”. This supports targeting repetitive rechecking rather than indiscriminately modifying arithmetic computation.

I propose one direction-specific intervention: remove the mean “Wait”/“ Wait” readout direction from the last six MLP outputs and last two full-attention outputs. Each writer’s orthogonal output components remain unchanged. This attempts to weaken reinforcement of the observed loop prefix; the supplied evidence does not establish a causal head or guarantee improvement.

```json
{
  "edits": [
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.18.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.19.self_attn.o_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.19.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.20.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.21.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.22.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.23.self_attn.o_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    },
    {
      "op": "project_out",
      "tensor": "model.language_model.layers.23.mlp.down_proj.weight",
      "direction": {
        "type": "unembed_mean",
        "tokens": ["Wait", " Wait"],
        "fold_final_norm": true
      },
      "strength": 1.0,
      "cols": "all"
    }
  ]
}
```