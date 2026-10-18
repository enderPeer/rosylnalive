# ADR 0000: How LRM records decisions

- **Status:** accepted, 2026-10-18.
- **Owner:** S0.
- **Scope:** every architecture decision record in `docs/decisions/`.

## Context

The plan has many agents making decisions in parallel. Routine engineering choices are reported in one line in a PR body. A few decisions bind other workstreams or reverse an earlier one, and those need a durable record that later agents can find and cite. Examples are a binding recipe, a toolchain pin, or a tolerance changed with evidence.

## Decision

- **What gets an ADR.** A decision gets an ADR when it does one of these:
  - binds another workstream;
  - changes a contract's meaning;
  - pins a toolchain or an external fact;
  - records a measured result that later work relies on.

  Routine choices stay in the PR body.
- **Where ADRs live.** `docs/decisions/`, under names that `docs/ownership.json` assigns:
  - `0000-*.md` and `0001-*.md`: cross-cutting records owned by S0. This file is the process, and `0001-shader-toolchain.md` is the toolchain;
  - `<family>-<slug>.md`: a family's own records, for example `R1-vk-heap-recipe.md`, `R5-warp.md` or `M1-regex.md`.
- **Format.** Every ADR has the same parts:
  1. a title `# ADR <id>: <decision in a few words>`;
  2. a short header: status (proposed, accepted, superseded by …), date, owner and scope;
  3. **Context**: the problem and the constraints;
  4. **Decision**: what is decided, as statements a reviewer can check;
  5. **Evidence**: what was measured or verified, with exact commands, versions, hashes and the machine. A claim without evidence is marked as unverified;
  6. **Consequences**: who must act, and what would reopen the decision;
  7. **Alternatives**: what was considered, and why it was not chosen.
- **ADRs are immutable once accepted.** Typos may be fixed. A changed decision gets a new ADR that supersedes the old one, and the old one's status line points to it.
- **ADRs and contracts.** An ADR that changes a contract cites the contract-change PR and its CHANGELOG entry (OwnershipAndContractChange). The contract text stays the normative rule, and the ADR records why it holds.

## Consequences

Agents cite ADRs by file name in PRs, contracts and code comments. The integrator checks, at review, that any binding decision in a PR has an ADR, and the hygiene ownership check keeps each family inside its own ADR names.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| [0000](0000-adr-process.md) | How LRM records decisions | accepted |
| [0001](0001-shader-toolchain.md) | Shader toolchain and bindless source convention | accepted; its SPIR-V binding part is provisional until R1's ADR and R4b |
