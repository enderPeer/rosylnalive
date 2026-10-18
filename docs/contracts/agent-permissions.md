# AgentPermissions contract

Owner: S0 (S0a; enforcement in hygiene is S0b's). Status: v1. Consumers: every agent session.

- **No committed allowlist.** AP-0 was declined by the integrator on 2026-10-17, so no `.claude/settings.json` is committed. Agent permissions stay the user's own settings, and workflow agents run under the session's permission mode.
  - The hygiene size check enforces this: `.claude/settings.json` and `.claude/settings.local.json` are forbidden files in any letter case (`tools/ci/hygiene.json`).
  - If AP-0 is ever approved, the integrator removes that rule in the same PR that adds the allowlist.
- **Integrator-only commands.** Only the integrator runs merge, tag and branch-protection commands. Agents never merge PRs, push to `main`, tag, or change repository settings.
- **PR labels and integrator branches.** An agent may put `contract-change` on its own PR. Only the integrator puts `ownership-exception` on a PR or opens an `int/` branch. In CI the hygiene check accepts either only from a login listed under `integrators` in `docs/ownership.json`, and lists every use in its summary (OwnershipAndContractChange).
- **Pushing.** Agents push only their own branch (`git push -u origin <branch>`, or `--force-with-lease` after a rebase).
- **Downloads.** Agents download only what an approval point lists (see `docs/dev-plan.md`, section 9). Everything else is reported to the integrator as a request.
- **System settings.** No agent changes system or security settings, including NVIDIA driver updates.
