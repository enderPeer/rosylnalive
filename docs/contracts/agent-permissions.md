# AgentPermissions contract

Owner: S0 (S0a). Status: v1. Consumers: every agent session.

- **No committed allowlist.** AP-0 was declined by the integrator on 2026-10-17, so no `.claude/settings.json` is committed. Agent permissions stay the user's own settings, and workflow agents run under the session's permission mode.
- **Integrator-only commands.** Only the integrator runs merge, tag and branch-protection commands. Agents never merge PRs, push to `main`, tag, or change repository settings.
- **Pushing.** Agents push only their own branch (`git push -u origin <branch>`, or `--force-with-lease` after a rebase).
- **Downloads.** Agents download only what an approval point lists (see `docs/dev-plan.md`, section 9). Everything else is reported to the integrator as a request.
- **System settings.** No agent changes system or security settings, including NVIDIA driver updates.
