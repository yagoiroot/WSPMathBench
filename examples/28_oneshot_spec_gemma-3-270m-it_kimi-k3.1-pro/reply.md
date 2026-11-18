The failures suggest two recurring output habits rather than a single repairable arithmetic head. First, the model often enters a terminal **“Final Answer:”** pathway and stops without the requested `\boxed{}`. Second, its rigid **“Let … be …”** setup frequently introduces variable-binding errors: quantities are dropped, reassigned, or attached to the wrong entity.

I therefore edit only late residual-writing projections. The `Final` projections remove the component by which these writers directly support that sentence-initial scaffold. They are weaker in earlier layers and strongest in the last MLP, with additional late-attention edits so attention cannot immediately restore the same component. This leaves lowercase `final answer`, `\boxed`, the embeddings, and the output layer untouched; it changes a token-specific pathway rather than overall sharpness. Existing calculations can therefore retain their result while the instructed boxing pathway faces less competition.

Two much weaker `Let` projections nudge the model away from compulsory symbolic recoding and toward the direct quantity-by-quantity calculations visible in the reference solutions. I avoid guessing that an unobserved head is an “arithmetic” or “repetition” head, and leave QK normalization and global-context retrieval intact.

```json
{"edits":[
  {"op":"project_out","tensor":"model.layers.14.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":0.4,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.15.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":0.55,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.16.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":0.8,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.17.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":1.5,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.16.self_attn.o_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":0.35,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.17.self_attn.o_proj.weight","direction":{"type":"unembed_token","token":"Final","fold_final_norm":true},"strength":0.65,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.16.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Let","fold_final_norm":true},"strength":0.3,"cols":"all"},
  {"op":"project_out","tensor":"model.layers.17.mlp.down_proj.weight","direction":{"type":"unembed_token","token":"Let","fold_final_norm":true},"strength":0.5,"cols":"all"}
]}
```