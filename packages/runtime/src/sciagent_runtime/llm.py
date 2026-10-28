"""LLM client protocol and deterministic scripted test client."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

import httpx
from pydantic import BaseModel

from .budget import (
    RequestOutcome, current_budget, current_budget_status, execution_budget, invoke_model,
)
from .compaction import compaction_input, compaction_input_text
from .context import ContextBudgetExceeded, ContextComposer
from .models import ComposedContext, ContextSection, ToolCallTurn
from .cancellation import OperationCancelled, check_cancelled, current_cancellation
from .http import send_request
from ._trace import write_trace_record
from .tool_calling import (
    NativeToolCallError, native_actions, native_input, parse_tool_turn,
)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Configured context limits for one injected model client."""

    context_window: int
    reserved_output_tokens: int = 4096
    safety_margin_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.context_window < 1 or self.reserved_output_tokens < 1:
            raise ValueError("context window and output reservation must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety margin cannot be negative")
        if (
            self.reserved_output_tokens + self.safety_margin_tokens
            >= self.context_window
        ):
            raise ValueError("model profile leaves no room for input context")

    def input_budget(
        self,
        *,
        schema_tokens: int,
        component_limit: int,
    ) -> int:
        """Return the smaller of the model capacity and component policy."""

        if schema_tokens < 0:
            raise ValueError("schema_tokens cannot be negative")
        if component_limit < 1:
            raise ValueError("component_limit must be positive")
        available = (
            self.context_window
            - self.reserved_output_tokens
            - self.safety_margin_tokens
            - schema_tokens
        )
        if available < 1:
            raise ValueError("model profile leaves no room after action schema")
        return min(component_limit, available)


class LLMClient(Protocol):
    """Native tool-calling client; Session owns the paired protocol history.

    Budget and trace hooks are optional. A minimal client consumes one model
    request per call and provides a stable protocol identity for continuation.
    """

    @property
    def tool_session_key(self) -> str: ...

    def next_tool_call(
        self, context: ComposedContext, schemas: list[dict],
        turns: list[ToolCallTurn], *, max_input_tokens: int,
    ) -> ToolCallTurn: ...


class StructuredLLMClient(Protocol):
    """Structured output for stateless callers such as Workflow Compiler."""

    def next_action(
        self, context: ComposedContext, action_type: type[BaseModel],
    ) -> BaseModel | dict: ...


class PromptLLMClient:
    """Adapt plain prompts to budgeted context without running an AgentLoop.

    The caller supplies its own system prompt and section label. Optional
    provider hooks retain the same semantics as direct runtime-client use.
    """

    manages_usage = True

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        system_prompt: str,
        max_context_tokens: int,
        section_name: str = "request",
        output_tool_name: str | None = None,
    ) -> None:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self._client = client
        self._system_prompt = system_prompt
        self._max_context_tokens = max_context_tokens
        self._section_name = section_name
        self._output_tool_name = output_tool_name
        self._composer = ContextComposer()
        self.last_attempts = 0
        self._attempt_limit = 3

    def set_trace_context(self, **kwargs) -> None:
        tracer = getattr(self._client, "set_trace_context", None)
        if tracer is not None:
            tracer(**kwargs)

    def set_attempt_limit(self, max_attempts: int) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._attempt_limit = max_attempts
        limiter = getattr(self._client, "set_attempt_limit", None)
        if limiter is not None:
            limiter(max_attempts)

    def next_action(
        self, prompt: str, action_type: type[BaseModel],
    ) -> BaseModel | dict:
        # A budget failure makes no provider call, even after a previous retry.
        self.last_attempts = 0
        budgeter = getattr(self._client, "context_budget", None)
        max_tokens = (
            budgeter(action_type, self._max_context_tokens)
            if budgeter is not None and self._output_tool_name is None else self._max_context_tokens
        )
        context = self._composer.compose(
            self._system_prompt,
            [ContextSection(
                name=self._section_name, content=prompt,
            )],
            max_tokens=max_tokens,
        )
        with execution_budget(
            max_llm_calls=self._attempt_limit,
            timeout_seconds=getattr(self._client, "timeout_seconds", 120),
        ) as budget:
            before = budget.usage.used
            context = context.model_copy(update={"runtime_status": current_budget_status()})
            try:
                estimated = self._composer.estimate_tokens(
                    "\n\n".join((context.text, context.runtime_status)))
                if estimated > max_tokens:
                    raise ContextBudgetExceeded("prompt and runtime status exceed the input budget")
                context = context.model_copy(update={"estimated_tokens": estimated})
                if self._output_tool_name is None:
                    return invoke_model(self._client, "next_action", context, action_type)
                context = context.model_copy(update={"protocol_mode": "output"})
                schemas = [{"type": "function", "function": {
                    "name": self._output_tool_name,
                    "description": "Return the requested structured result using this sole output tool.",
                    "parameters": action_type.model_json_schema(),
                }}]
                turn = invoke_model(self._client, "next_tool_call", context, schemas, [],
                                    max_input_tokens=self._max_context_tokens)
                if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != self._output_tool_name:
                    raise NativeToolCallError(f"Expected exactly one {self._output_tool_name} output call")
                return action_type.model_validate_json(turn.tool_calls[0].arguments)
            finally:
                self.last_attempts = budget.usage.used - before
                self._attempt_limit = 3


class LLMExhaustedError(RuntimeError):
    """Raised when a scripted client has no action left to return."""


class LLMTextResponseError(ValueError):
    """Raised when a compaction response is not complete plain text."""


class ScriptedLLMClient:
    """Deterministic native client returning predefined tool-call turns."""

    tool_session_key = "scripted-native-tools/v1"

    def __init__(self, turns: list[ToolCallTurn]) -> None:
        self._turns = deque(ToolCallTurn.model_validate(turn) for turn in turns)
        self.contexts: list[ComposedContext] = []
        self.schemas: list[list[dict]] = []
        self.histories: list[list[ToolCallTurn]] = []

    def next_tool_call(
        self, context: ComposedContext, schemas: list[dict],
        turns: list[ToolCallTurn], *, max_input_tokens: int,
    ) -> ToolCallTurn:
        """Record current context and return the next native model reply."""
        self.contexts.append(context)
        self.schemas.append(schemas)
        self.histories.append([turn.model_copy(deep=True) for turn in turns])
        if not self._turns:
            raise LLMExhaustedError("scripted LLM has no remaining turn")
        return self._turns.popleft()


class OpenAICompatibleClient:
    """Stateless transport for native tools or Compiler structured JSON output.

    The caller owns native conversation history; sharing a client cannot share
    Session reasoning or tool results. Neither path falls back to the other.
    """

    manages_usage = True

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key_env: str,
        model_profile: ModelProfile | None = None,
        timeout_seconds: int = 120,
        trace_dir: Path | None = None,
        trace_level: str = "off",
    ) -> None:
        self.model = model
        self.endpoint = f"{api_base.rstrip('/')}/chat/completions"
        self.api_key_env = api_key_env
        self.model_profile = model_profile
        self.timeout_seconds = timeout_seconds
        if trace_level not in {"off", "metadata", "full"}:
            raise ValueError("trace_level must be off, metadata or full")
        if trace_level != "off" and not trace_dir:
            raise ValueError("trace_dir is required when trace_level is enabled")
        self.trace_dir = Path(trace_dir).expanduser().resolve() if trace_dir else None
        self.trace_level = trace_level
        self._trace_context: dict = {}
        self._trace_seq = 0
        self._last_call_id: str | None = None
        self.last_attempts = 0
        self._attempt_limit: int | None = None

    @staticmethod
    def _action_instruction(action_type: type[BaseModel]) -> str:
        schema = json.dumps(action_type.model_json_schema(), ensure_ascii=False)
        return (
            "Return exactly one JSON object matching this action schema. "
            "Do not use markdown fences.\n"
            f"{schema}"
        )

    def context_budget(
        self,
        action_type: type[BaseModel],
        component_limit: int,
    ) -> int:
        """Return this component's usable input budget for one action schema."""

        if self.model_profile is None:
            return component_limit
        schema_tokens = ContextComposer.estimate_tokens(
            self._action_instruction(action_type)
        )
        try:
            return self.model_profile.input_budget(
                schema_tokens=schema_tokens,
                component_limit=component_limit,
            )
        except ValueError as error:
            raise ContextBudgetExceeded(str(error)) from error

    def set_attempt_limit(self, max_attempts: int) -> None:
        """Limit provider attempts for the next call only."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._attempt_limit = max_attempts

    def tool_input_limit(self, component_limit: int) -> int:
        """Native schema/history are counted in the total request, not twice."""
        if self.model_profile is None:
            return component_limit
        return self.model_profile.input_budget(schema_tokens=0, component_limit=component_limit)

    @property
    def tool_session_key(self) -> str:
        """Bind private continuation to the same wire protocol, endpoint and model."""
        identity = json.dumps(["openai-compatible-tools/v2", self.endpoint, self.model])
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def next_tool_call(
        self,
        context: ComposedContext,
        schemas: list[dict],
        turns: list[ToolCallTurn],
        *,
        max_input_tokens: int,
    ) -> ToolCallTurn:
        """Request native calls; serial execution remains the Runtime's policy."""
        self.last_attempts = 0
        included = [*context.included_sections, "native_tools"]
        if turns:
            included.append("tool_history")

        def prepare():
            current = context.model_copy(update={"runtime_status": current_budget_status(),
                                                  "included_sections": included})
            wire_input = native_input(current, schemas, turns)
            message = json.dumps(wire_input, ensure_ascii=False)
            estimated = ContextComposer.estimate_tokens(message)
            if estimated > self.tool_input_limit(max_input_tokens):
                raise ContextBudgetExceeded("native request exceeds the total input budget")
            current = current.model_copy(update={"estimated_tokens": estimated})
            body_data = {"model": self.model, **wire_input, "temperature": 0}
            if self.model_profile is not None:
                body_data["max_tokens"] = self.model_profile.reserved_output_tokens
            return current, message, body_data

        current, message, body_data = prepare()
        # No JSON response_format, force-tool option or beta strict mode. The
        # regular native protocol is shared across OpenAI-compatible providers.
        return self._request(current, message, body_data, native=True, prepare=prepare)

    def summarize_history(
        self,
        prompt: str,
        *,
        max_input_tokens: int,
    ) -> str:
        """Make one bounded plain-text checkpoint request through shared transport."""

        self.last_attempts = 0
        if max_input_tokens < 1:
            raise ValueError("compaction input budget must be positive")

        def prepare():
            status = current_budget_status()
            message = compaction_input_text(prompt, runtime_status=status)
            estimated = ContextComposer.estimate_tokens(message)
            if estimated > self.tool_input_limit(max_input_tokens):
                raise ContextBudgetExceeded("compaction request exceeds the total input budget")
            context = ComposedContext(
                text=prompt, runtime_status=status, included_sections=["compaction"],
                omitted_sections=[], estimated_tokens=estimated,
            )
            body_data = {
                "model": self.model, **compaction_input(prompt, runtime_status=status), "temperature": 0,
            }
            if self.model_profile is not None:
                body_data["max_tokens"] = self.model_profile.reserved_output_tokens
            return context, message, body_data

        context, message, body_data = prepare()
        return self._request(context, message, body_data, native=False, text=True, prepare=prepare)

    def set_trace_context(self, **kwargs) -> None:
        """Attach per-call correlation fields for the optional JSONL trace."""
        self._trace_context = dict(kwargs)

    def _write_trace(self, record: dict) -> None:
        """Append one JSONL trace line (the API key is never recorded)."""
        if self.trace_dir is None or self.trace_level == "off":
            return
        self._trace_seq += 1
        write_trace_record(self.trace_dir, {"sequence": self._trace_seq, **record})

    def record_validation(self, validation_error: str) -> None:
        """Record caller-side candidate validation, keyed by the last call_id."""
        if self.trace_dir is None or self.trace_level == "off":
            return
        self._write_trace(
            {
                "call_id": self._last_call_id,
                "schema_validation_error": validation_error,
            }
        )

    @staticmethod
    def _sha256(text: str | None) -> str | None:
        if text is None:
            return None
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _response_trace_fields(
        self,
        parsed_action,
        raw_response_text,
        raw_reasoning_text: str | None,
        validation_error: str | None,
        usage,
        finish_reason: str | None,
        response_message: dict | None = None,
        *,
        text: bool = False,
    ) -> dict:
        """Project one response with the same content boundary at every attempt."""
        if not text and isinstance(parsed_action, ToolCallTurn):
            try:
                actions = native_actions(parsed_action)
                parsed_action = actions[0] if len(actions) == 1 else actions
            except (json.JSONDecodeError, NativeToolCallError) as error:
                parsed_action = None
                validation_error = str(error)
        tool = (
            parsed_action.get("tool")
            if not text and isinstance(parsed_action, dict)
            else None
        )
        record = {
            "tool": tool,
            "action_valid": None if text else parsed_action is not None,
            "validation_error": validation_error,
            "usage": usage,
            "finish_reason": finish_reason,
        }
        if isinstance(parsed_action, list):
            record["tools"] = [action["tool"] for action in parsed_action]
        if self.trace_level == "full":
            record["raw_response_text"] = raw_response_text
            record["parsed_action"] = None if text else parsed_action
            record["raw_reasoning_text"] = raw_reasoning_text
            if response_message is not None and "tool_calls" in response_message:
                record["raw_tool_calls"] = response_message["tool_calls"]
        else:
            record["response_sha256"] = self._sha256(raw_response_text)
            record["action_sha256"] = (
                self._sha256(json.dumps(parsed_action, ensure_ascii=False, default=str))
                if not text and parsed_action is not None
                else None
            )
            if response_message is not None and "tool_calls" in response_message:
                record["tool_calls_sha256"] = self._sha256(response_message["tool_calls"])
        return record

    def _trace_record(
        self,
        context: ComposedContext,
        message: str,
        started: float,
        call_id: str,
        created_at: str,
        request_max_tokens: int | None,
        attempts: list[dict],
    ) -> dict:
        """Keep one logical-call row; response fields describe its last attempt.

        Bounded attempt details retain earlier failures without multiplying
        logical-call rows or changing retry accounting. A null output limit
        means the request omitted max_tokens, not that the provider is unlimited.
        """
        record = {**self._trace_context, **attempts[-1]}
        record.update({
            "call_id": call_id,
            "created_at": created_at,
            "model": self.model,
            "included_sections": context.included_sections,
            "omitted_sections": context.omitted_sections,
            "estimated_tokens": context.estimated_tokens,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "request_max_tokens": request_max_tokens,
            "attempts": attempts,
        })
        if self.trace_level == "full":
            record["request_text"] = message
        else:
            record["request_sha256"] = self._sha256(message)
        return record

    def next_action(
        self,
        context: ComposedContext,
        action_type: type[BaseModel],
    ) -> BaseModel | dict:
        """Structured output for Compiler and explicit JSON-only callers."""
        def prepare():
            current = context.model_copy(update={"runtime_status": current_budget_status()})
            message = "\n\n".join(part for part in (
                current.text, current.runtime_status, self._action_instruction(action_type),
            ) if part)
            current = current.model_copy(update={
                "estimated_tokens": ContextComposer.estimate_tokens(message),
            })
            if (self.model_profile is not None
                    and current.estimated_tokens > self.model_profile.input_budget(
                        schema_tokens=0, component_limit=self.model_profile.context_window)):
                raise ContextBudgetExceeded("structured request exceeds the total input budget")
            body_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": message}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }
            if self.model_profile is not None:
                body_data["max_tokens"] = self.model_profile.reserved_output_tokens
            return current, message, body_data

        current, message, body_data = prepare()
        return self._request(current, message, body_data, native=False, prepare=prepare)

    def _request(
        self,
        context: ComposedContext,
        message: str,
        body_data: dict,
        *,
        native: bool,
        text: bool = False,
        prepare: Callable[[], tuple[ComposedContext, str, dict]] | None = None,
    ):
        """One transport/retry/accounting implementation for both output protocols."""
        if native and text:
            raise ValueError("native tool and plain-text response modes are exclusive")
        attempt_limit = min(3, self._attempt_limit or 3)
        self._attempt_limit = None
        self.last_attempts = 0
        with execution_budget(
            max_llm_calls=attempt_limit,
            timeout_seconds=self.timeout_seconds * attempt_limit + attempt_limit - 1,
        ):
            return self._request_attempts(
                context, message, body_data,
                native=native, text=text, attempt_limit=attempt_limit, prepare=prepare,
            )

    def _request_attempts(
        self, context, message, body_data, *, native, text, attempt_limit, prepare=None,
    ):
        budget = current_budget()
        assert budget is not None
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable {self.api_key_env}")
        call_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        self._last_call_id = call_id
        started = time.monotonic()
        last_error: Exception | None = None
        parsed_action = None
        attempts: list[dict] = []
        try:
            for attempt in range(attempt_limit):
                budget.check()
                if attempt:
                    wait = budget.remaining_timeout(1.0)
                    cancellation = current_cancellation()
                    if cancellation is not None:
                        cancellation.wait(wait)
                    else:
                        time.sleep(wait)
                check_cancelled()
                if prepare is not None:
                    prepared_context, prepared_message, prepared_body = prepare()
                else:
                    prepared_context, prepared_message, prepared_body = context, message, body_data
                request = httpx.Request(
                    "POST", self.endpoint,
                    content=json.dumps(prepared_body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
                budget.charge(call_id, attempt)
                # Preparation can outlive the deadline or fail to persist a charge.
                # Only a reserved attempt owns this input in the response trace.
                context, message, body_data = prepared_context, prepared_message, prepared_body
                self.last_attempts = attempt + 1
                # Never attribute an earlier response to a later transport failure.
                raw_response_text = None
                raw_reasoning_text = None
                response_message = None
                parsed_action = None
                usage = None
                finish_reason = None
                last_error = None
                outcome: RequestOutcome = "unknown"
                try:
                    timeout = budget.remaining_timeout(self.timeout_seconds)
                    response = send_request(request, timeout=timeout, max_response_bytes=None)
                    outcome = "failed"
                    payload = json.loads(response.read().decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise TypeError("provider response must be an object")
                    # Capture provider diagnostics before parsing the action JSON.
                    usage = payload.get("usage")
                    choice = payload["choices"][0]
                    if not isinstance(choice, dict):
                        raise TypeError("provider choice must be an object")
                    finish_reason = choice.get("finish_reason")
                    response_message = choice["message"]
                    if not isinstance(response_message, dict):
                        raise TypeError("provider message must be an object")
                    reasoning_content = response_message.get("reasoning_content")
                    raw_reasoning_text = (
                        reasoning_content if isinstance(reasoning_content, str) else None
                    )
                    raw_response_text = response_message.get("content")
                    if not native and not text and not isinstance(raw_response_text, str):
                        raise TypeError(
                            "provider returned non-string message content: "
                            f"{type(raw_response_text).__name__}"
                        )
                    content = raw_response_text.strip() if isinstance(raw_response_text, str) else ""
                    if not native and not text and content.startswith("```"):
                        content = content.removeprefix("```json").removeprefix("```")
                        content = content.removesuffix("```").strip()
                except OperationCancelled as error:
                    last_error = error
                    raise
                except httpx.DecodingError as error:
                    outcome = "failed"
                    last_error = error
                    raise
                except httpx.HTTPStatusError as error:
                    detail = error.response.content[:2000].decode("utf-8", errors="replace")
                    outcome = "failed"
                    last_error = error
                    # Preserve the existing no-retry policy for client errors.
                    if error.response.status_code < 500:
                        last_error = RuntimeError(f"LLM HTTP {error.response.status_code}: {detail}")
                        raise last_error from error
                except (
                    httpx.TransportError, TimeoutError, json.JSONDecodeError,
                    KeyError, IndexError, TypeError,
                ) as error:
                    last_error = error
                    budget.remaining_timeout()
                else:
                    # The provider envelope was valid. Malformed model output
                    # needs caller feedback, not an identical HTTP retry. Parse
                    # outside the transport handlers so the error reaches the
                    # caller, while both finally blocks still preserve trace.
                    try:
                        if native:
                            parsed_action = parse_tool_turn(response_message, finish_reason)
                        elif text:
                            if finish_reason != "stop":
                                raise LLMTextResponseError(
                                    "compaction response did not finish with stop"
                                )
                            if response_message.get("tool_calls"):
                                raise LLMTextResponseError(
                                    "compaction response must not contain tool calls"
                                )
                            if not isinstance(raw_response_text, str) or not content:
                                raise LLMTextResponseError(
                                    "compaction response must contain non-empty text"
                                )
                            parsed_action = content
                        else:
                            parsed_action = json.loads(content)
                        outcome = "succeeded"
                    except (
                        json.JSONDecodeError, NativeToolCallError, LLMTextResponseError,
                    ) as error:
                        last_error = error
                        raise
                    break
                finally:
                    budget.usage.complete(call_id, attempt, outcome)
                    attempts.append({
                        "retry_number": attempt,
                        "estimated_tokens": context.estimated_tokens,
                        **({"request_text": message} if self.trace_level == "full" else
                           {"request_sha256": self._sha256(message)}),
                        **self._response_trace_fields(
                            parsed_action, raw_response_text, raw_reasoning_text,
                            str(last_error) if last_error is not None else None,
                            usage, finish_reason,
                            response_message,
                            text=text,
                        ),
                    })
        finally:
            if attempts:
                self._write_trace(self._trace_record(
                    context, message, started, call_id, created_at,
                    body_data.get("max_tokens"), attempts,
                ))
        if last_error is not None:
            raise RuntimeError(
                f"LLM request failed after {self.last_attempts} attempts: {last_error}"
            ) from last_error
        return parsed_action
