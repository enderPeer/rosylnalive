# O1-0001: Disabling EOS in generate() needs a neutral model generation config

Status: accepted (O1, 2026-10-18). Affects: ReferenceOracles (greedy rules), G2, and anyone who calls `generate()` on an oracle.

## Context

The plan (section 6) says `generate()` gets a fresh `GenerationConfig(do_sample=False, max_new_tokens=64, eos_token_id=None)`, and must give 64 tokens that equal the manual loop and a no-cache recompute.

In transformers 5.20.1, `GenerationMixin._prepare_generation_config` does not use the passed config as given:
1. It copies the config.
2. It calls `update(**self.generation_config.to_dict(), defaults_only=True)`, which overwrites every field that is still `None` with the model's own generation config.

`eos_token_id=None` is indistinguishable from "unset", so the checkpoint's EOS comes back:
- Qwen3-0.6B: `[151645, 151643]`, from `generation_config.json`;
- the tiny model: its `config.json` value, through `GenerationConfig.from_model_config`.

It was measured on the tiny model. With the plan's recipe alone, `generate()` stops after 10 tokens, at the configured EOS (token 7, which greedy emits at step 9), while the manual loop runs 64 steps.

## Decision

`common.greedy_generate` swaps `model.generation_config` for a neutral `GenerationConfig()` for the duration of the call and restores it afterwards. It also:
- asserts the effective config, read back through `_prepare_generation_config`: `eos_token_id` None, `do_sample` false, no `min_new_tokens`;
- records the effective config in the manifest;
- runs `greedy_generate_without_neutral_config` as a control on every tiny greedy prompt, and records that it stops at EOS.

`min_new_tokens` stays unused, as the plan says.

## Consequences

- G2 must use `common.greedy_generate`. The plan's one-line recipe would make the "manual loop = generate()" check fail on any prompt whose greedy output contains EOS within 64 steps.
- **The tiny model's EOS rule.** The tiny model's `eos_token_id` is chosen as a token that greedy decoding of p0 actually emits (at step 9). So a decoder that honours EOS cannot match the tiny greedy golden. This gives M2a's `Greedy.run(…, StopAtEos = false)` a real test.
- **Requested plan change.** The integrator may update the plan's wording in section 6 to point here.
