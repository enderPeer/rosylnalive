# LRM — Large Roslyn Model: Design

17 October 2026 · exported from the shared design doc

## Summary

The LRM (Large Roslyn Model) is a transformer whose tokens, structure and coherence come from Roslyn. It reads and writes C# and IL as Roslyn objects. Its experts are calls to Roslyn's public API, made from inside its layers.

**Thesis.** An LLM learns code from text and checks nothing. A compiler checks everything but understands no intent. The LRM keeps the compiler's exact structure in the loop and learns only what the compiler cannot supply: intent, choice and planning.

**Goals**

- One shared vocabulary for C# syntax, IL and Roslyn results, with each symbol identified exactly across all of them.
- Coherence enforced by Roslyn during generation, not checked afterwards.
- The output is a Roslyn object (syntax tree, compilation, assembly) that forms itself through mutations, not text.
- Many valid emission forms are accepted, because one IL has many C# translations and vice versa.
- C# and IL experts cooperate, each changing only its own artifact.

**Non-goals for now:** natural-language chat, and languages other than C# and IL. The system is designed for networked GPU clusters from the start.

**First success criterion:** at matched compute, a model with Roslyn layers beats the strongest baseline on functional correctness, measured as pass@k on hidden tests. That baseline is a code model with Roslyn-constrained decoding, a compile-and-test repair loop, and RL from compiler and test feedback. Compile rate and reference resolution are reported but not scored, because pointer identifiers make them high by construction.

**Roadmap:** phase 1 runs a small Qwen model forward on our stack, and phase 2 trains it. Together they prove the binding and core system. Phase 3 is the LRM itself.

## Background

The design rests on a Roslyn research bundle from 17 October 2026. It has three runnable demos built on Roslyn 5.9.0, ILSpy 11.1.0.9782 and .NET SDK 10.0.402.

| Finding | Evidence | Consequence for the LRM |
| --- | --- | --- |
| Source round-trips exactly through tokens and trivia | 15 tokenization checks passed, including on malformed input | Text is a reliable boundary, and Roslyn can represent partial programs |
| Compiler operations can be requested on demand | Parsing, symbols, binding, diagnostics and emission were each selected at runtime | Each operation can be an expert |
| IL is resolved: `call 0x06000001` names exactly one method | The `Run` bytes `181928010000062A` decoded to `Demo.Add` | Symbol identity can be exact, like metadata tokens |
| Decompiling keeps behaviour, not form | Results 5, 5, then 7 after an edit; a comment was lost and expression bodies became blocks | IL to C# is many-to-many, so the loss must accept equivalents |

**Why LLMs fall short for code.** LLM tokens split text by frequency, carry no kind or value, and never resolve to symbols. Generation is append-only and unchecked, so a plausible wrong name (`Sum` for `Add`) passes silently. Context is a flat window where contradictions coexist and stale facts stay. A compiler is the opposite: closed, scoped and immutable, and every contradiction is an error.

## Vocabulary and symbol identity

Every unit the LRM reads or writes is a Roslyn-defined token, and every reference resolves to one shared symbol ID.

| Token class | Examples | Defined by |
| --- | --- | --- |
| Syntax | `IdentifierToken`, `InvocationExpression`, trivia kinds | `SyntaxKind` |
| IL | `ldc.i4.2`, `call`, `ret` | The IL opcode set |
| Symbol reference | `sym#12` | The symbol table of the current compilation |
| Result | `[DIAG CS7036]`, `[TYPE int]`, `[SCOPE ...]` | Expert outputs |
| Lexical | Sub-word pieces of names, comments and commit messages | A small text sub-vocabulary |

**Shared symbol identity.** `sym#12` is one embedding wherever `Add` appears: in source, in IL (`call 0x06000001` resolved to `sym#12`), in diagnostics, in goals and in commit messages. Code, IL and results become one language instead of three.

**Identifiers as pointers.** At each position the model points to a candidate symbol, or declares a new name spelled in lexical tokens, so every reference is in scope by construction. `LookupSymbols` alone does not cover every case, so candidates are built in steps:

- Names in scope: `LookupSymbols(position)`, plus labels from `LookupLabels`.
- Members after a dot: `LookupSymbols(position, container: receiverType)`, with `includeReducedExtensionMethods` for extension methods.
- Types not yet imported: candidates from referenced assemblies, chosen with an "add using" action.
- Overloads and generics: a method group is resolved by a second pointer step over its overloads, then over type arguments.

**Lexical channel.** The meaning of names and comments is English, so the LRM keeps a sub-word channel. Lexical tokens become facts about symbols that guide routing. Roslyn alone decides correctness.

**Symbol embeddings are composed, not memorized.** Symbol IDs are local to one compilation, like metadata tokens. A symbol's embedding is therefore built from its facts: kind, type, signature, containing type and name pieces. A newly declared symbol with few facts starts from its name and declared type.

**Literals.** Numeric literals are spelled as digit tokens plus a value channel taken from `ValueText`. String and character literals use lexical tokens. Interpolated strings alternate lexical text with ordinary expression syntax.

## Architecture

Roslyn commands are the experts of a mixture-of-experts. A router inside the layers picks commands, their results return as tokens, and mutations change the program being built.

| Expert kind | Examples | Effect | Calls per forward pass |
| --- | --- | --- | --- |
| Query | `LookupSymbols`, `GetSymbolInfo`, `GetTypeInfo`, `GetDiagnostics`, `AnalyzeDataFlow`, IL inspection | Returns facts | Many, in parallel |
| Mutation | `SyntaxFactory` constructors, `ReplaceNode`, `InsertNodesAfter`, compile C# to IL, decompile IL to C# | Produces a new immutable snapshot | Few, in sequence |
| Content | Names, docs, usages (`SymbolFinder.FindReferencesAsync`), commit history, strings | Text read by an encoder and linked to symbols | As needed |

**Roslyn layers.** Every few transformer layers, a Roslyn layer gathers queries from the whole batch, groups them by expert, runs them and writes the results back. Results enter in two ways:

- As embeddings added to the residual stream: kind, symbol, type, constant value, diagnostics.
- As learned attention biases on edges: parent and child, use to declaration, data flow, control flow.

**Temporary snapshots.** A mutation first applies to a temporary snapshot. Later layers query it and then commit or discard it; immutability makes this cheap and gives free undo. Inside a method body, Roslyn's speculative semantic model binds new statements and expressions without a new compilation. It cannot add members that other code can see, so new helper stubs go through `Compilation.ReplaceSyntaxTree`. Usage facts from `SymbolFinder.FindReferencesAsync` need a Workspaces solution and are slow, so they are precomputed.

**Differences from a standard MoE.** The experts are exact and have no weights, and their outputs cannot be averaged. Cheap, deterministic query experts are not routed at all: their facts are computed for every position. Only expensive experts are routed, with a load-balancing term plus a cost penalty, so the router cannot collapse onto a few experts.

```text
+------------------+          +----------------------+          +----------------------------+
| Context regions  |  read    | Transformer layers   |          | Roslyn experts, in-process |
|  Goal            | -------> | attention + MLP      |          |  Query experts             |
|  Program  <------+--+       +----------+-----------+  route   |  Mutation experts          |
|  Knowledge       |  |                  v              +------> |  Content experts           |
|  Results  <---+  |  |       +----------------------+  |       +-------------+--------------+
+---------------+--+  |       | Roslyn layer: router |--+                     |
                |     |       +----------------------+                        |
                |     | commit +----------------------+   new snapshot        |
                |     +------- | Temporary snapshot   | <---------------------+
                |              | commit or discard    |                       |
                |              +----------------------+                       |
                +------------- answers as versioned tokens <------------------+
```

The router sends each call to a Roslyn expert. Answers return to Results as versioned tokens, and committed snapshots replace the Program.

## Workspace and context

The context is four typed regions in one shared vocabulary. Every result is tagged with the snapshot version it came from.

| Region | Contents |
| --- | --- |
| Goal | Stubs, typed holes and tests (they compile to IL), plus intent text |
| Program | The evolving syntax tree and its emitted IL |
| Results | Expert outputs as tokens, tagged by snapshot version |
| Knowledge | Names, docs, usages and commits linked to symbols |

**Goals as code.** A goal is a partial program: method stubs, typed holes such as `HOLE<int>@IL`, and tests. Roslyn reports the expected type at each hole, so subgoals can be checked. Breaking a goal down means writing helper stubs. The plan is a call graph that compiles to IL, so the model's own thinking is re-embedded as IL tokens. A text goal sits beside the code goal for intents that types cannot express, such as "readable" or "faster".

**Snapshot discipline.** When the program changes, results from older snapshots are masked. This applies Roslyn's immutable-snapshot rule to the model's own context and keeps stale facts out.

**Reasoning over results.** The model predicts each expert's answer before calling it. The prediction loss makes it internalize Roslyn. The surprise between prediction and answer tells the router when a call is worth making.

## Backward pass

Roslyn is physics, not a parameter: gradients flow around it, never through it.

| Kind of backward | What flows back | Across | Mechanism |
| --- | --- | --- | --- |
| Gradient | Derivatives | Neural layers within one pass | Backpropagation; the residual bypasses Roslyn within one snapshot; effects of committed mutations get no gradient |
| Estimated gradient | A substitute derivative | Discrete choices of expert and arguments | Gate scaling, top-k mixtures, straight-through via a learned predictor |
| Credit | Targets and rewards | Committed mutations and whole episodes | Imitation, repair targets, RL with a critic |

**Gate scaling.** Each expert's output is scaled by its router probability, y = p_i × Enc_i(Roslyn_i(S, a)). The router learns through p_i, as in the Switch Transformer.

**Straight-through via learned Roslyn.** The forward pass uses Roslyn's real answer. The backward pass uses the gradient of the model's predicted answer (sg = stop-gradient):

```latex
e = \mathrm{Enc}(f_{\text{real}}) + \mathrm{Enc}(\hat f(h)) - \mathrm{sg}\big[\mathrm{Enc}(\hat f(h))\big]
```

Values stay exact, and gradients follow the model's own understanding of Roslyn. Two rules make this sound:

- **The predictor learns only from its prediction loss.** The task loss is stopped before the weights of f̂. Otherwise f̂ learns facts that help the loss instead of true facts, and its gradients mislead.
- **This gradient updates belief, not choice.** It tells the model how to change what it expects, not which discrete argument to pick. Expert arguments are trained with teacher labels where they exist, and otherwise with a score-function (policy-gradient) estimator against a learned baseline.

**Diagnostics as repair targets.** A diagnostic names a location, not a cause. CS7036 fires at the call site even when the fix belongs in the declaration, and one error can have many valid fixes. Diagnostics are therefore not gradients. They generate repair training data (error, snapshot, a fix that removes it) and they shape rewards. ILVerify errors play the same role for IL.

**Total loss**

```latex
L = L_{\text{imitation}} + \lambda_1 L_{\text{predict}} + \lambda_2 L_{\text{repair}} + \lambda_3 L_{\text{RL}} + \lambda_4 L_{\text{cost}}
```

Pretraining uses only the first two terms on precomputed facts, so it is nearly ordinary supervised training. Rewards measure progress (holes filled, tests passing), not raw error counts, because partial programs are supposed to have errors.

## Training

Training reads masses of C# and IL, aligns the two, and then climbs a curriculum in which the C# and IL experts must cooperate.

**Pretraining**

1. C#: repositories that build. Roslyn facts are computed on prefixes. Objectives: predict the next action, masked nodes and expert answers.
2. IL: NuGet packages, plus IL compiled from the C# corpus under several settings. Objectives: predict opcodes, stack types, token resolution and ILVerify results.
3. Alignment: PDB sequence points map IL offsets to source spans. Objectives: predict IL from C# and back, and rank equivalent decompilations. The alignment is coarse and many-to-many: sequence points are per statement, async methods, iterators and lambdas move code into generated types, and many packages ship without PDBs.

**Linearization.** Finished files hold end states, not edits. Each file is turned into one canonical mutation sequence: namespace and type declarations, then every member signature, then method bodies in syntax-tree pre-order. Git history adds real edit orders as a second source.

**Prefix scoping.** C# members can be used before they are declared. Declaration-level facts (types and member signatures) therefore come from the whole file's skeleton, which matches generation because signatures are written first. Body facts come from prefixes.

**Cooperative environment.** The C# agent owns `.cs` snapshots and the IL agent owns `.dll` snapshots; neither can write the other's. The compile and decompile bridges have a cost and are disabled at some levels. Signatures that cross the boundary are contract tokens in the goal region. Success is a shared reward, and each side also gets its own checks (Roslyn diagnostics, ILVerify). Both sides are one policy with shared weights, so cooperation is enforced by these rules rather than expected to emerge.

| Level | Goal | Why both sides are needed |
| --- | --- | --- |
| 1 | Hello world | C# `Main` calls `Greeting()` from an IL-authored assembly |
| 2 | Values and arithmetic | IL computes and C# formats |
| 3 | Control flow | C# runs the loop and IL implements the predicate |
| 4 | Calls in both directions | Contracts are negotiated across TypeRef and MemberRef |
| 5 | Types | IL defines types that C# subclasses or implements |
| 6 | Translation and two-key edits | Both artifacts are edited independently and must behave the same |
| 7 | Repair | Bugs sit in a DLL with no source and in its C# caller |
| 8 | IL-only abilities | Return-type overloads, `tail.` calls, fault handlers |
| 9 | C#-only practicality | IL weaves into async and LINQ output without touching the source |
| 10 | Real projects | Commits are the goals, and the model chooses the split |

**Task sources:** procedural generators, hindsight relabeling (cutting holes into real programs), bug injection, splitting programs and discarding part of the source, and git history. Sample tasks the model sometimes solves, and replay earlier levels.

**Isolation.** Building the corpus runs untrusted code: MSBuild tasks, analyzers and source generators. Corpus builds and every generated program therefore run under OS isolation (a container or VM) with no network, CPU, memory and time limits, and a fresh environment per job. A separate process or a collectible `AssemblyLoadContext` is not a sandbox.

**Data licensing.** Only permissively licensed repositories are used, with the license recorded per file. NuGet packages are decompiled only where their license allows it.

## Evaluation, baselines and kill criteria

The LRM is judged on functional correctness, at matched compute, against the strongest baseline, with stop rules fixed before the first run.

| Item | Choice |
| --- | --- |
| Primary metric | pass@1 and pass@10 on hidden tests |
| Function-level benchmark | MultiPL-E, C# subset |
| Repo-level benchmark | Held-out C# repositories, with tasks from commits dated after the training cutoff and deduplicated against the corpus |
| Baseline | Same size and compute: Roslyn-constrained decoding, a compile-and-test repair loop, RL from compiler and test rewards |
| Ablations | No Roslyn layers; query experts only; no goal region |
| Reported, not scored | Compile rate, reference resolution, expert calls per task, wall-clock time per task |

**Kill criteria (proposed; to be fixed before the first run)**

- Stop the in-layer design if, at two model sizes, Roslyn layers gain less than 2 percentage points pass@1 over the baseline on the repo-level benchmark.
- Stop RL on a curriculum level if the joint pass rate does not improve over three successive evaluations.
- Drop an expert if disabling it changes no scored metric.

**Estimates still needed:** model sizes and training tokens, the compute to compute facts on prefixes, the share of baseline failures that Roslyn can detect, and the CPU cores per GPU needed so that Roslyn never stalls the GPU.

## Systems

Everything is written in F#, Roslyn runs in the same process as the model, and both GPU backends share one bindless design.

- **Stack:** an F# model, autodiff tape and optimizer; Roslyn in-process on worker threads; a small F# hardware interface over D3D12 and Vulkan; shaders written once in HLSL (or Slang) and compiled to DXIL and SPIR-V.
- **Vulkan:** `VK_EXT_descriptor_heap` is required, and devices without it are rejected. We bind Vulkan with our own F# P/Invoke binding to the C API.
- **Heap layout:** one resource heap, written once at startup. It holds a few arena buffers (weights, activations, gradients, optimizer, Roslyn in, Roslyn out), a tensor table and a symbol table. Kernels receive 32-bit tensor tokens in 256 bytes of push data (`vkCmdPushDataEXT`) or root constants.
- **Autodiff:** a discriminated union of graph nodes. `RoslynHop` is one case, and its backward rule is the straight-through estimator. A CPU reference implementation with finite-difference checks is the oracle for every GPU kernel.

| Concept | D3D12 | Vulkan |
| --- | --- | --- |
| Heap | Shader-visible CBV/SRV/UAV heap | Resource heap |
| Writing a descriptor | `CreateUnorderedAccessView` | `vkWriteResourceDescriptorsEXT` with an address range |
| Kernel arguments | Root constants | `vkCmdPushDataEXT` |
| Pipeline | PSO plus one shared root signature | `DESCRIPTOR_HEAP_BIT` with a `NULL` layout |
| CPU-GPU sync | `ID3D12Fence` | Timeline semaphore |

**Measured limits.** Our device check accepts an RTX 4080, and its `maxPushDataSize` is exactly 256 bytes, the minimum the extension allows. The proposal recommends 512 so that tools have room, so we cap kernel arguments at 128 bytes. On D3D12, 64 root constants fill the whole 64-DWORD root signature, which works only with directly indexed heaps (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`).

**Speed.** Pretraining uses precomputed facts, so Roslyn is not in that loop at all. During RL and inference:

- Roslyn sync points every few layers.
- Queries dispatched per expert, like MoE all-to-all.
- Micro-batches that overlap GPU and CPU work.
- Speculative semantic models for temporary edits.
- A content-addressed cache over immutable snapshots.
- The model's own prediction as a fast path, verified in the background.

Sync points are a fence protocol. The GPU writes queries and signals; the CPU answers and signals back; command lists recorded in advance wait on that signal.

**Networked clusters from the start.** The system is built for clusters of GPU nodes from day one.

- Each node runs its GPU workers and a pool of Roslyn workers on its own CPUs, so Roslyn calls never cross the network.
- Data parallelism across nodes; expert and tensor parallelism inside a node.
- Vulkan and D3D12 have no NCCL equivalent, so the project builds its own collective layer in F#: ring and tree all-reduce over RDMA where the network supports it and TCP otherwise, overlapped with the backward pass.
- Episodes, snapshots and cached Roslyn facts are content-addressed, so any node can serve or reuse them.
- Reductions use a fixed order and fixed seeds, so multi-node runs can be reproduced.

**Robust to unstable networks.** Links can lag, drop or partition, and nodes can disappear. Results stay exact anyway, because every piece of work can be recomputed anywhere.

- **Work units are pure and content-addressed.** A unit (one micro-batch's gradient, an RL episode, a batch of Roslyn facts) is a deterministic function of its inputs: weights version, data shard ID and seed. Any node can recompute any unit, and duplicates are harmless.
- **Recompute on lag.** Each unit has a deadline set from recent completion times. A unit that misses it is reissued to another node, and the first result wins.
- **Reduce by shard, not by node.** Gradients are summed in a fixed order of shard IDs. When a node drops, its shards are recomputed elsewhere, so the global batch and the result stay the same.
- **Tight inside a node, loose across nodes.** Expert and tensor parallelism stay on a node's reliable links. Only data-parallel gradient exchange and work units cross the network.
- **Membership epochs.** A replicated coordinator tracks live nodes with heartbeats and leases. Each step runs under one membership epoch, and during a partition only the majority side continues.
- **Replicated state.** Weight and optimizer shards live on at least two nodes and are checkpointed asynchronously, so losing a node never loses state.
- **Stale RL rollouts.** Episodes carry the policy version that produced them. Rollouts older than a set bound are dropped or reweighted.
- **Chaos tests.** Test runs delay, drop and kill nodes at random. The final weights must match a failure-free run bit for bit.

## Traps and design principles

Most ways this can fail come from breaking one of a few simple rules. The traps come first, then the principles that avoid them.

| Trap | Why it bites | Avoid it by |
| --- | --- | --- |
| Computing training facts on finished files | The model learns to rely on facts about code it has not written yet, which it never has at generation time | Declaration facts from the file's skeleton, body facts from prefixes; generate declarations before bodies |
| A learned embedding per symbol ID | IDs are local to one compilation, like metadata tokens, so a per-ID embedding memorizes noise | Compose each symbol's embedding from its Roslyn facts: kind, type, signature, name pieces |
| Letting the task loss train the predictor | f̂ learns convenient facts instead of true ones, and its gradients mislead | Train f̂ only on its prediction loss |
| Treating diagnostics as gradients | A diagnostic names a location, not a cause, and has many valid fixes | Use diagnostics as repair data and reward shaping |
| Exposing the whole Roslyn API as experts | Argument spaces explode and nothing composes | A small algebra of single-step mutations that compose, the way IL's opcodes do |
| One canonical target per program | Valid alternative forms are penalized, and the model collapses onto one decompiler style | Set-valued targets where equivalents can be listed; test-based rewards where they cannot |
| Rewarding fewer errors at every step | Necessary intermediate states are punished (`Add()` before its arguments exist) | Reward progress: holes filled, tests passing, at checkpoints |
| Free compile and decompile bridges | The C# agent compiles everything and cooperation never forms | Bridges with a cost and per-level availability, enforced by the environment |
| Gameable goals | The model hard-codes the expected output, edits tests or emits unverifiable IL that confuses the checker | Hidden tests, held-out inputs, property tests, mandatory ILVerify, tests outside the agents' ownership |
| Metrics met by construction | Pointer identifiers make reference resolution and compile rate look great without proving anything | Score functional correctness against the strongest baseline |
| Building the corpus on a bare machine | Builds run untrusted MSBuild tasks, analyzers and source generators | OS isolation for every build and run |
| Roslyn in the pretraining loop | GPU throughput collapses waiting on the CPU | Precompute every fact for known code |
| Results from old snapshots left in context | The model reasons over stale facts, the classic LLM failure | Tag results by snapshot version and mask stale ones |
| Letting the lexical channel decide correctness | LLM fuzziness returns through names and comments | Text guides routing; only Roslyn and execution decide |
| Nondeterministic GPU or network reductions | Runs cannot be reproduced and bugs cannot be bisected | Fixed reduction order, sort-based scatter-add, fixed seeds |

**Design principles**

1. **Everything is a token into a table.** IL metadata tokens, `sym#`, tensor tokens and descriptor indices are all small integers that point to rows. One mechanism serves the compiler, the model and the GPU.
2. **Everything is immutable.** Snapshots, the startup heap and versioned results give undo, branching, caching by content hash and staleness checks for free.
3. **Roslyn is physics.** Never learn what Roslyn computes exactly. Learn only what it cannot supply, which is intent, choice and planning, and let gradients flow around it.
4. **One loop at three scales: predict, act, compare.** Predict-then-call inside a layer, a critic predicting each mutation's effect, and tests at the end of an episode. Surprise drives both learning and routing.
5. **Goals are programs.** The goal, the plan, progress and reward are all expressed and checked in the same language the model writes.
6. **Teacher first, reward last.** Exact Roslyn labels carry most of training. RL covers only what labels cannot express.
7. **The bridge is the curriculum.** The cost and availability of the compile and decompile bridges is the one setting that shapes cooperation from level 1 to level 10.
8. **The CPU reference is the specification.** Every GPU kernel and both backends are judged against the same F# code.

## Risks and open questions

The largest risk is that Roslyn layers do not beat the strongest baseline on functional correctness. The kill criteria decide this before further building.

| Risk | Mitigation |
| --- | --- |
| Roslyn layers give little gain over the strongest baseline | Ablation at two sizes, judged by the written kill criteria |
| Most failures are logic errors that Roslyn cannot see | Goals as code (tests, holes) carry the logic signal; measure early what share of baseline failures Roslyn can detect |
| Router collapse onto a few experts | Cheap experts computed every time; balancing term plus cost penalty for routed ones |
| The learned predictor drifts during RL and gives bad gradients | Train it only on its prediction loss, on the model's own outputs; track its accuracy |
| The model learns to ignore expert results | Ablate experts and compare against runs with them disabled |
| CPU Roslyn stalls the GPU | Precompute, batch, pipeline and cache; Roslyn workers on every node; measure cost tiers early |
| Our own collectives are slow without NCCL | Overlap all-reduce with the backward pass; measure bandwidth per node early |
| Context regions overflow | Short result tokens, masking of stale results, per-symbol summaries |
| A hand-built GPU stack is slow at first | Tune matmul and attention first; use cooperative-matrix support where drivers expose it |
| `VK_EXT_descriptor_heap` support varies by driver | The RTX 4080 passes our device check; start with the binding-mapping path, not direct heap indexing |

**Open questions**

- [ ] How are new identifiers spelled: pointers to `LookupSymbols` candidates plus lexical tokens for new names?
- [ ] What loss accepts every equivalent emission form when the equivalents cannot be listed?
- [ ] How do the C# and IL experts split work when no rule forces it?
- [ ] How often can real goals be written as code (stubs, holes, tests)?

## Milestones

Two proof-of-concept phases run a small Qwen model on our own stack, forward and then backward. They prove the F# binding and the core system before LRM work starts.

```text
+-----------------------+          +-----------------------+          +-----------------------+
| Phase 1: Qwen forward |   Gate   | Phase 2: Qwen backward|   Gate   | Phase 3: LRM          |
| safetensors+tokenizer |--> 1 --> | autodiff tape + AdamW |--> 2 --> | Roslyn facts+experts  |
| attention,RMSNorm,RoPE|          | data-parallel training|          | ablation vs baseline  |
| D3D12+Vulkan, 2 nodes |          | fault-tolerance       |          | C# and IL curriculum  |
+-----------------------+          +-----------------------+          +-----------------------+
   (current)          logits and tokens        gradients match;
                      match the reference      chaos run identical
```

Phase 1 is current. Each gate must pass before the next phase starts; the full criteria are listed below.

**Phase 1: Qwen forward pass.** Run a small Qwen model (Qwen3-0.6B) for inference entirely on our stack.

- F# loaders for the safetensors weights and the model's byte-level BPE tokenizer.
- Kernels for embedding, RMSNorm, rotary embeddings, grouped-query attention, the SwiGLU MLP and the KV cache, with architecture details read from the model's config.
- An F# CPU reference of the forward pass as the oracle for every kernel.
- Both backends, then the same forward pass split across two nodes.
- **Gate 1:** on both backends, logits match golden outputs from the official reference implementation within tolerance, and greedy decoding produces identical tokens.

**Phase 2: Qwen backward pass.** Train the same model on our stack.

- Backward kernels, the autodiff tape, AdamW and mixed precision, checked against the CPU reference with finite differences.
- Data-parallel training across nodes using the fault-tolerance protocol.
- **Gate 2:** one-step gradients match reference gradients within tolerance, a short fine-tune lowers the loss, and a chaos run ends with weights identical to a failure-free run.

**Phase 3: LRM.** Starts once both gates pass.

1. **Toy Roslyn layer:** a `RoslynHop` calling real Roslyn, with a test confirming that the task loss never reaches the predictor's weights.
2. **Fact pipeline:** canonical linearization; declaration facts from each file's skeleton and body facts from prefixes; builds inside isolated sandboxes; the 15 round-trip checks as regression tests.
3. **Evaluation harness:** MultiPL-E C# and the held-out repo benchmark, hidden tests, the strongest baseline, and the kill criteria fixed in writing.
4. **Ablation:** Roslyn layers against the baseline at two model sizes, judged by the kill criteria.
5. **Level 1 environment harness:** a C# snapshot plus an IL-authored assembly, contracts, an isolated run, and scripted agents playing both sides.
6. **Mutation experts and the curriculum:** imitation on linearized traces, then RL through the ten levels, with test and compile rewards shaped by repair targets.

## First-principles review

An independent reviewer found the design coherent but the evidence plan weak. Every finding marked "Agree" below has now been applied in the sections above.

| Severity | Finding | Response |
| --- | --- | --- |
| Critical | The success criterion is circular: pointer identifiers make reference resolution near 100% by design, and constrained decoding raises compile rate with no training | Agree. Switch to functional correctness (pass@k on hidden tests) at matched compute |
| Critical | The strongest baseline is missing: a code LLM with Roslyn-constrained decoding, a compile-and-test repair loop and RL from compiler and test feedback | Agree. Add it, with written kill criteria |
| Critical | The thesis assumes many failures are ones the compiler can detect. Most modern code-LLM failures are logic errors, which Roslyn cannot see | Partly agree. The thesis rests on goals as code (tests, holes) as much as on compiler facts; measure both |
| Critical | The task loss must not update the predictor f̂, or it learns facts that help the loss rather than true facts | Agree. Train f̂ only on its prediction loss |
| Critical | Mutation sequences cannot be read off finished files; no canonical order is defined | Agree. Define a linearization (declarations first, then pre-order bodies); git history adds real edit orders |
| Critical | Building the corpus runs untrusted MSBuild tasks, analyzers and generators; a separate process is not a sandbox | Agree. Build and run inside OS isolation (container or VM), no network, resource limits |
| Major | Prefix facts break on members declared later in the file, teaching "declare new" for symbols that exist | Agree. Take declaration-level facts from the whole file's member skeleton, and body facts from prefixes |
| Major | Straight-through gradients update the model's belief, not its discrete action; arguments still need a policy-gradient estimator | Agree. State the estimator for arguments explicitly |
| Major | Replacing load balancing with a cost penalty makes router collapse more likely | Agree. Compute all cheap query experts every time; route only expensive ones, with a balancing term |
| Major | Diagnostics name a location, not a cause, and have many valid fixes; they are shaping or repair data, not gradients | Agree. Rename them repair targets and use them as data and shaping |
| Major | `LookupSymbols` misses members after a dot, extension methods, unimported types and overload choice | Agree. The pointer vocabulary needs receiver-type lookup, extension methods and an overload step |
| Major | Speculative semantic models cannot add new members; `FindReferencesAsync` needs a Workspaces `Solution` and is slow | Agree. Helper stubs use `ReplaceSyntaxTree`; usage facts are precomputed |
| Major | Missing: benchmarks, kill criteria, compute estimates, data licensing, literal tokenization | Agree. Add each before building |
| Minor | PDB alignment is coarse (statement level, generated types for async and lambdas, packages without PDBs) | Accept as many-to-many alignment |

**Recommendations that conflicted with earlier decisions**

- **Test the thesis on a pretrained PyTorch model first: declined.** F# and the descriptor-heap GPU stack are used from the start.
- **Cut IL authoring and the cooperative environment from phase 1: resolved by the roadmap.** Phases 1 and 2 are the Qwen proof of concept; IL authoring and cooperation belong to phase 3.

Running the Qwen proof of concept on our own stack also tests the reviewer's concern about hand-built kernel speed early, before any LRM-specific work.

**Vulkan status as reported by the reviewer:** `VK_EXT_descriptor_heap` shipped in Vulkan 1.4.340 (January 2026). Reported driver support: NVIDIA 610 and later; Intel ANV on by default from Mesa 26.2; AMD RADV opt-in in Mesa 26.1; VKD3D-Proton behind a flag. Sources: [proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_descriptor_heap.adoc), [NVIDIA](https://developer.nvidia.com/blog/streamlining-resource-binding-with-end-to-end-support-for-vulkan-descriptor-heaps/), [RADV](https://www.phoronix.com/news/RADV-Merges-Descriptor-Heap), [ANV](https://phoronix.com/news/Intel-ANV-Descriptor-Heaps-On), [VKD3D-Proton](https://phoronix.com/news/VKD3D-Proton-Descriptor-Heaps).
