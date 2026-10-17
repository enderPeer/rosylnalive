# LRM — Large Roslyn Model

A transformer whose tokens, structure and coherence come from Roslyn. It reads and writes C# and IL as Roslyn objects, and its experts are calls to Roslyn's public API made from inside its layers. Written in F#, on its own GPU stack for Direct3D 12 and Vulkan (`VK_EXT_descriptor_heap`), designed for networked GPU clusters.

- [Design document](docs/LRM-design.md)

## Layout

| Path | Contents |
| --- | --- |
| `docs/LRM-design.md` | Architecture, backward pass, training curriculum, systems, evaluation, roadmap and review |
| `gpu/vulkan` | F# P/Invoke binding to the Vulkan C API and a device check that rejects devices without `VK_EXT_descriptor_heap` |
| `roslyn-research-2026-10-17.zip` | Earlier Roslyn research: notes plus compiler-pipeline, tokenization and IL round-trip demos |

## Roadmap

1. **Phase 1:** run Qwen3-0.6B forward on our stack. Gate: logits and greedy tokens match the reference.
2. **Phase 2:** train it (backward pass, multi-node, fault tolerant). Gate: gradients match; a chaos run gives identical weights.
3. **Phase 3:** the LRM itself.

## Device check

Requires the .NET 10 SDK.

```bash
dotnet run --project gpu/vulkan/app/VkDeviceCheck
```
