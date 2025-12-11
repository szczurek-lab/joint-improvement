"""Self-improvement stage for offline optimisation."""

from __future__ import annotations

from .pipeline import (
    SelfImprovementConfig,
    SelfImprovementResult,
    run_self_improvement,
)

__all__ = [
    "SelfImprovementConfig",
    "SelfImprovementResult",
    "run_self_improvement",
]
