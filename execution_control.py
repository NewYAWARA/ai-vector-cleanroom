"""Fail-closed execution budgets shared by the CLI and local workbench.

The cooperative checks in this module make candidate boundaries cancellable
and keep progress evidence deterministic.  They are deliberately paired with
the workbench's per-job subprocess supervisor: native NumPy, renderer, or
tracer calls cannot be safely interrupted from another Python thread.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional


SCHEMA = "aivc-execution-control-v1"


class ConversionInterrupted(RuntimeError):
    """Base class for terminal, fail-closed execution interruptions."""

    terminal_status = "failed"


class ConversionCancelled(ConversionInterrupted):
    terminal_status = "cancelled"


class ConversionTimedOut(ConversionInterrupted):
    terminal_status = "timed_out"


class CandidateBudgetExceeded(ConversionInterrupted):
    terminal_status = "budget_exhausted"


class ExecutionControl:
    """Track one conversion's monotonic deadline, cancellation, and candidates."""

    def __init__(
        self,
        *,
        budget_seconds: Optional[float] = None,
        candidate_cap: int = 16,
        cancel_requested: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget_seconds is not None:
            budget_seconds = float(budget_seconds)
            if not math.isfinite(budget_seconds) or budget_seconds <= 0.0:
                raise ValueError("budget_seconds must be a positive finite number")
        if isinstance(candidate_cap, bool) or int(candidate_cap) < 1:
            raise ValueError("candidate_cap must be a positive integer")
        self.budget_seconds = budget_seconds
        self.candidate_cap = int(candidate_cap)
        self._cancel_requested = cancel_requested or (lambda: False)
        self._clock = clock
        self.started_at = float(clock())
        self.deadline_at = (
            self.started_at + budget_seconds
            if budget_seconds is not None else None
        )
        self.stage = "queued"
        self.candidate_planned = 0
        self.candidate_started = 0
        self.candidate_evaluated = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self.started_at)

    def set_candidate_plan(self, planned: int) -> None:
        if isinstance(planned, bool) or int(planned) < 0:
            raise ValueError("planned candidate count must be non-negative")
        self.candidate_planned = int(planned)
        if self.candidate_planned > self.candidate_cap:
            raise CandidateBudgetExceeded(
                f"required candidate plan {self.candidate_planned} exceeds "
                f"the fail-closed cap {self.candidate_cap}")

    def checkpoint(self, stage: Optional[str] = None) -> None:
        if stage:
            self.stage = str(stage)
        if bool(self._cancel_requested()):
            raise ConversionCancelled(
                f"conversion cancelled during {self.stage}")
        if (self.deadline_at is not None
                and float(self._clock()) >= self.deadline_at):
            raise ConversionTimedOut(
                f"conversion exceeded its {self.budget_seconds:g}-second "
                f"budget during {self.stage}")

    def before_candidate(self, signature=None) -> None:
        self.checkpoint("candidate_search")
        if self.candidate_started >= self.candidate_cap:
            raise CandidateBudgetExceeded(
                f"candidate cap {self.candidate_cap} exhausted before the "
                "required search completed")
        self.candidate_started += 1

    def after_candidate(self) -> None:
        self.candidate_evaluated += 1
        self.checkpoint("candidate_search")

    def snapshot(self) -> dict:
        return {
            "schema": SCHEMA,
            "stage": self.stage,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "budget_seconds": self.budget_seconds,
            "candidate_evaluated": self.candidate_evaluated,
            "candidate_started": self.candidate_started,
            "candidate_planned": self.candidate_planned,
            "candidate_cap": self.candidate_cap,
        }


__all__ = [
    "SCHEMA",
    "CandidateBudgetExceeded",
    "ConversionCancelled",
    "ConversionInterrupted",
    "ConversionTimedOut",
    "ExecutionControl",
]
