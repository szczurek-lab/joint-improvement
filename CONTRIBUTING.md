# Contributing Guidelines

## Setup

Run the setup script (creates environment, installs dependencies, and sets up pre-commit hooks):

```bash
scripts/setup_env.sh
```

## Pre-commit Hooks

Hooks run automatically on `git commit`. Check before committing:

```bash
pre-commit run
```

**Fixing ruff issues:**
```bash
ruff check --fix src/joint_improvement  # Auto-fix linting
ruff format                              # Format code
```

## Workflow

1. Open an issue describing the change
2. Create branch from `main`
3. Make commits (run `pre-commit run` before each commit)
4. Before pushing: `pre-commit run --all-files` (includes pytest)
5. **Rebase on `main` BEFORE pushing:** `git rebase main` (keeps history clean)
6. Push branch: `git push origin <branch-name>`
7. Open PR (link to issue)
8. Keep PRs focused (<300 LOC)
