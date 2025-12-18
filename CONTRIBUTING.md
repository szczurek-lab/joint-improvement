# Contributing Guidelines

## Setup (recommended)

Run the setup script (creates environment, installs dependencies, and installs pre-commit hooks):

```bash
scripts/setup_env.sh
```

If you prefer manual setup, ensure you have a Python environment with the dev extras installed:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Code quality checks

### Pre-commit hooks
Hooks run automatically on `git commit` for staged files. To run them manually:

```bash
pre-commit run
```

To run all hooks on all files (slower; includes pytest):

```bash
pre-commit run --all-files
```

### Ruff (lint + format)
Ruff is the primary linter/formatter for this repo.

```bash
ruff check --fix src/joint_improvement   # Auto-fix lint issues in src
ruff format .                            # Format code (repo-wide)
```

### mypy (optional, non-blocking)
We keep mypy lightweight and scoped to `src/joint_improvement`.

Run via pre-commit (manual stage; does not run on normal commits):

```bash
pre-commit run mypy --all-files
```

Or run directly:

```bash
python -m mypy src/joint_improvement
```

### Tests
Run the test suite:

```bash
pytest
```

## Notebooks
Notebooks are cleaned by `nbstripout` via pre-commit. If you commit notebooks, run:

```bash
pre-commit run nbstripout
```

## Workflow / PRs

1. Open an issue describing the change
2. Create branch from `main`
3. Make commits (run `pre-commit run` before each commit)
4. Before pushing: `pre-commit run --all-files` (includes pytest)
5. **Rebase on `main` BEFORE pushing:**
    ```
    git fetch origin
    git rebase origin/main
    # resolve conflicts if needed
    git push origin my-feature-branch
    ```
6. Push branch: `git push origin <branch-name>`
7. Open PR (link to issue)
8. Keep PRs focused (<300 LOC)
