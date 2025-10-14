"""Engine-side interventions: how the sandbox engine weights and orders sources in the prompt.

These change nothing on the page. They model a search engine that ranks the target a little
higher (`target_boost`) or places it in a given slot (`target_at`), which is how the sandbox
measures sensitivity to retrieval weighting and context order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Order = Literal["relevance", "reverse", "random", "target_at"]


@dataclass(frozen=True)
class RetrievalPolicy:
    order: Order = "relevance"
    target_position: int | None = None  # 1-based slot for order="target_at"
    target_boost: float = 0.0
    boost_stage: Literal["pre", "post"] = "post"

    @property
    def name(self) -> str:
        parts = [self.order if self.order != "target_at" else f"target_at:{self.target_position}"]
        if self.target_boost:
            boost = f"boost:{self.target_boost:g}@{self.boost_stage}"
            parts = [boost] if self.order == "relevance" else [*parts, boost]
        return "+".join(parts)

    @classmethod
    def parse(cls, spec: str) -> RetrievalPolicy:
        """'relevance', 'reverse', 'random', 'target_at:2', 'boost:0.05', 'boost:0.05@pre'."""
        order: Order = "relevance"
        pos = None
        boost = 0.0
        stage: Literal["pre", "post"] = "post"
        for part in spec.split("+"):
            if part.startswith("target_at:"):
                order, pos = "target_at", int(part.split(":")[1])
            elif part.startswith("boost:"):
                val = part.split(":", 1)[1]
                if "@" in val:
                    val, st = val.split("@")
                    stage = "pre" if st == "pre" else "post"
                boost = float(val)
            elif part in ("relevance", "reverse", "random"):
                order = part  # type: ignore[assignment]
            else:
                raise ValueError(f"bad retrieval policy {spec!r}")
        return cls(order, pos, boost, stage)


RELEVANCE = RetrievalPolicy()
