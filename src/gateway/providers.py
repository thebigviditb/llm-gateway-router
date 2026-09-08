"""Model provider adapters.

A provider turns a CompletionRequest into a CompletionResult or raises one of the
narrow ProviderError subclasses below. Nothing else may escape `complete()`; the
router relies on that to decide failover and to keep upstream detail out of client
responses.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

log = logging.getLogger("gateway.providers")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1)
    system: str | None = Field(default=None, max_length=50_000)

    def prompt_text(self) -> str:
        return (self.system or "") + "".join(m.content for m in self.messages)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProviderError(Exception):
    """Upstream failed. `detail` is log-only and must never reach a client."""

    def __init__(self, provider: str, detail: str, status: int | None = None):
        super().__init__(f"{provider}: status={status} {detail}")
        self.provider, self.detail, self.status = provider, detail, status


class ProviderRateLimited(ProviderError):
    def __init__(self, provider: str, detail: str = "429 Too Many Requests"):
        super().__init__(provider, detail, status=429)


class Provider(Protocol):
    name: str

    async def complete(self, req: CompletionRequest) -> CompletionResult: ...


# ---------------------------------------------------------------------------------
# Mock provider (tests + local demo)
# ---------------------------------------------------------------------------------
Behavior = Literal["ok", "429", "500", "hang", "crash"]


@dataclass
class MockProvider:
    name: str = "mock"
    model: str = "mock-model"
    behavior: Behavior = "ok"
    delay: float = 0.0          # seconds before responding (for "ok"/"429"/"500")
    output_tokens: int = 50
    calls: int = 0
    cancelled: int = 0
    completed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def complete(self, req: CompletionRequest) -> CompletionResult:
        self.calls += 1
        try:
            if self.behavior == "hang":
                await asyncio.sleep(3600)
            elif self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.behavior == "429":
            raise ProviderRateLimited(self.name, "upstream said: slow down (trace: 0xDEADBEEF)")
        if self.behavior == "500":
            raise ProviderError(self.name, "upstream stack trace: Exception at line 42", status=500)
        if self.behavior == "crash":
            raise RuntimeError("unexpected bug inside provider adapter")
        self.completed += 1
        prompt = req.prompt_text()
        return CompletionResult(
            text=f"[{self.name}] echo: {prompt[-60:]}",
            model=self.model,
            provider=self.name,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=min(self.output_tokens, req.max_tokens),
        )


# ---------------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------------
class AnthropicProvider:
    """Adapter over the official SDK.

    max_retries=0: the gateway owns retry/failover policy, so the SDK must not
    silently absorb 429s or stretch a call past the router's timeout.
    """

    def __init__(self, name: str, model: str, *, client=None, timeout: float = 30.0):
        import anthropic  # imported lazily so the mock path has no SDK dependency at runtime

        self.name = name
        self.model = model
        self._anthropic = anthropic
        self._client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=timeout)

    async def complete(self, req: CompletionRequest) -> CompletionResult:
        a = self._anthropic
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=req.max_tokens,
                messages=[m.model_dump() for m in req.messages],
            )
            if req.system:
                kwargs["system"] = req.system
            resp = await self._client.messages.create(**kwargs)
        except asyncio.CancelledError:
            raise
        except a.RateLimitError as e:
            raise ProviderRateLimited(self.name, f"{e.status_code} {e.message}") from e
        except a.APIStatusError as e:
            raise ProviderError(self.name, f"{e.status_code} {e.message}", status=e.status_code) from e
        except a.APIConnectionError as e:  # includes APITimeoutError
            raise ProviderError(self.name, f"connection error: {e}", status=None) from e
        except Exception as e:  # anything else is still an upstream failure from the router's view
            raise ProviderError(self.name, f"{type(e).__name__}: {e}", status=None) from e

        if resp.stop_reason == "refusal":
            raise ProviderError(self.name, "model refused the request", status=None)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return CompletionResult(
            text=text,
            model=resp.model,
            provider=self.name,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
