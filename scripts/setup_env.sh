#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-joint-improvement}

if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  micromamba create -f environment.yml -n "$ENV_NAME"
fi

# shellcheck disable=SC1091
micromamba activate "$ENV_NAME"

pip install -e .[dev]
pre-commit install
