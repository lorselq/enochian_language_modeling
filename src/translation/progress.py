from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .llm_synthesis import LLMRequestProgress, TranslationProgressReporter


class TranslationPhaseReporter(Protocol):
    """Generic reporter for translation pipeline phase updates."""

    def begin(
        self,
        phase: str,
        detail: str | None = None,
        total: int | None = None,
    ) -> None: ...

    def advance(
        self,
        phase: str,
        current: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> None: ...

    def end(self, phase: str, detail: str | None = None) -> None: ...


@dataclass(slots=True)
class LLMProgressReporterAdapter(TranslationProgressReporter):
    """Adapt legacy LLM request callbacks into generic phase callbacks."""

    reporter: TranslationPhaseReporter

    def prepare(self, progress: LLMRequestProgress) -> None:
        self.reporter.begin(
            "llm.prepare",
            detail="Preparing LLM synthesis",
            total=progress.total_primary_requests,
        )

    def start_primary(self, *, phase: str, current: int, total: int) -> None:
        self.reporter.advance(
            f"llm.{phase}",
            current=current,
            total=total,
            detail="primary request",
        )

    def start_validation(
        self,
        *,
        phase: str,
        current: int,
        total: int,
        kind: Literal["retry", "repair"],
    ) -> None:
        self.reporter.advance(
            f"llm.{phase}",
            current=current,
            total=total,
            detail=f"validation {kind}",
        )

    def done(self) -> None:
        self.reporter.end("llm", detail="LLM synthesis complete")
