"""Lossless composition of fixed context; only protocol history is compacted."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil

from .models import ComposedContext, ContextSection


DEFAULT_AGENT_CONTEXT_TOKENS = 256_000
CONTEXT_TARGET_SHARE = 0.80


class ContextBudgetExceeded(ValueError):
    """Required context cannot fit the configured input budget."""


class ContextComposer:
    """Keep every supplied section whole and measure the complete request."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """A deterministic character estimate, not a model tokenizer."""
        return 0 if not text else max(1, ceil(len(text) / 4))

    def compose(
        self,
        system_prompt: str,
        sections: list[ContextSection],
        *,
        max_tokens: int,
        measure: Callable[[str], int] | None = None,
    ) -> ComposedContext:
        if max_tokens < 1:
            raise ContextBudgetExceeded("max_tokens must be positive")
        names = [section.name for section in sections]
        if len(names) != len(set(names)):
            raise ValueError("context section names must be unique")
        rendered = [f"## {section.name}\n{section.content}" for section in sections]
        text = "\n\n".join([f"## system\n{system_prompt}", *rendered])
        tokens = (measure or self.estimate_tokens)(text)
        if tokens > max_tokens:
            raise ContextBudgetExceeded("required context exceeds the input budget")
        return ComposedContext(
            text=text, system_prompt=system_prompt,
            task_context="\n\n".join(
                part for section, part in zip(sections, rendered)
                if section.name not in {"material_index", "runtime_authority"}
            ),
            runtime_context="\n\n".join(
                part for section, part in zip(sections, rendered)
                if section.name == "runtime_authority"
            ),
            material_index="\n\n".join(
                part for section, part in zip(sections, rendered) if section.name == "material_index"
            ),
            included_sections=["system", *names], omitted_sections=[], estimated_tokens=tokens,
        )
