# Contributing Guidelines

This document describes the minimal workflow for parallel development.

## 1. Environment & Tooling
- Create the shared micromamba environment via `environment.yml` or `scripts/setup_env.sh`.
- Install development dependencies: `pip install -e .[dev]` inside the activated environment so local edits reflect immediately.
- Install `pre-commit` hooks: `pre-commit install` to run quality checks before every commit.

### Pre-commit Hooks
The repository uses pre-commit hooks to ensure code quality:
- **mypy**: Type checking for `joint_improvement` package
- **ruff-check**: Linting with auto-fix for `src/joint_improvement`
- **ruff-format**: Code formatting
- **pytest**: Run all tests
- **nbstripout**: Clean Jupyter notebooks (remove outputs and empty cells)

Run hooks manually: `pre-commit run --all-files`

## 2. Branching & Pull Requests
- Base all work off `main`; create feature branches named `feature/<topic>` or `fix/<topic>`.
- Keep PRs focused and small (prefer <300 LOC changed) to reduce merge conflicts.
- Rebase on `main` before opening a PR to ensure a linear history; resolve conflicts locally.
- Require approval from one teammate before merging; use GitHub's draft PRs for early feedback.

## 3. Testing & Quality Gates
- Run `pre-commit run --all-files` and `pytest` (including the tiny overfit smoke test) prior to pushing.
- For changes affecting Hugging Face integration, run `python -m joint_improvement.hf_trainer --config configs/hf_imdb.yaml` with the small debug config.

## 4. Communication & Reviews
- Keep PR descriptions concise but complete: motivation, key changes, validation steps.
- Use review comments to highlight assumptions or TODOs; follow up after merging.
- For breaking changes, outline a migration note in the PR and link it in README if user-facing.
