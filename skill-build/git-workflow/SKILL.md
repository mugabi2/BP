---
name: git-workflow
description: Manage safe, reviewable Git workflows for meaningful repository changes, from state inspection and branching through validation, commits, pushes, and pull requests. Use when implementing changes in a Git repository or when the user asks for Git delivery; exclude read-only reviews unless Git state is relevant.
---

# Git Workflow

Make Git history explain what was wrong, what changed, why, and how it was verified.

For meaningful work, use this lifecycle when the user has authorized end-to-end Git delivery:

**Issue -> Branch -> Changes -> Validation -> Commit -> Push -> Pull Request -> Human Review -> Merge**

Automatic skill selection is not permission for remote mutations. Create issues, push, or open pull requests only when the user explicitly requests end-to-end delivery or repository instructions clearly include those actions. Never merge without an explicit instruction.

## Inspect First

Before editing:

1. Run `git status --short` and identify the current branch.
2. Inspect remotes when push or pull-request work is in scope.
3. Check for an existing issue or branch that matches the task.
4. Treat pre-existing changes as user-owned and preserve them.
5. Avoid development edits directly on `main` or `master`; when authorized, use an appropriate working branch.

If unrelated changes overlap required files and cannot be separated safely, ask for direction.

## Issues and Branches

Use an issue for meaningful work when tooling supports it and issue creation is authorized. Bug fixes, features, schema or model changes, and significant configuration changes usually qualify; tiny spelling fixes usually do not.

Use a short, natural issue title. Keep its body concise:

```markdown
## Why
Explain the problem or reason.

## What needs to change
Describe the expected change plainly.

## Done when
State observable completion criteria.
```

Continue on the correct working branch when one exists. Otherwise, when authorized, use a short name:

- `fix/<issue>-<topic>`
- `feature/<issue>-<topic>`
- `change/<issue>-<topic>`
- `docs/<issue>-<topic>`

Include the issue number when available. Prepend any environment-required namespace, for example `codex/fix/27-paye-date-parsing`.

## Change and Validate

Change only what the task requires. Avoid unrelated rewrites, formatting churn, and speculative cleanup.

Before staging:

1. Review status and relevant diffs.
2. Run practical, proportionate tests, builds, or domain checks.
3. Confirm only intended files changed.
4. Fix failures caused by the work where practical and report unresolved failures.

Do not weaken validation merely to obtain a passing result.

## Commit

When committing is in scope, stage only task-related paths. Avoid blanket staging when unrelated changes exist.

Use a short, human-readable imperative message such as `Fix PAYE date parsing` or `Add Station dimension relationship`. Prefer one understandable unit per commit. Never commit credentials, private keys, or sensitive `.env` files.

## Push and Pull Request

Push only when remote delivery is authorized, authentication is already available, the intended remote is clear, and validation is acceptable.

- New branch: prefer `git push -u origin HEAD`.
- Tracked branch: use `git push`.
- Authentication failure: report it without placing credentials in repository content.

When authorized, open a pull request with a natural title and concise summary, changes, and validation sections. Include `Closes #<issue-number>` only when the PR should complete that issue. Leave the PR for human review unless explicitly asked to merge.

## Safety

Never automatically run `git push --force`, `git push --force-with-lease`, `git reset --hard`, `git clean -fd`, or delete branches with unmerged work. Do not discard user changes, bypass protections, rewrite shared history, or expose credentials. If destructive action appears necessary, stop and explain the situation and recovery options.

## Report

Report only items that actually exist: issue number/title, branch, commit message/hash, PR number/title/link, validation results, and unresolved problems or omitted lifecycle steps with their reasons.
