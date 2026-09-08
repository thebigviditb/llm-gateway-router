"""Primary -> secondary failover with a hard timeout race, wrapped in token accounting.

Failover triggers (spec): primary returns 429, or primary exceeds `primary_timeout`.
Extension: a non-429 upstream failure (5xx / connection error) also fails over, since
retrying the same request on another provider is the safe move in every such case.

Timeout mechanics
-----------------
`asyncio.timeout()` cancels the primary coroutine when the deadline passes and does not
return until that cancellation has been *processed*. So by the time we start the
secondary, the primary can no longer produce a result, and only one provider result can
ever reach the ledger commit. A late-arriving primary response is discarded by the
event loop, never by us.

Token accounting
----------------
reserve(estimate) -> call providers -> commit(actual) | release() on any failure.
commit/release run under asyncio.shield so a client disconnect (which cancels the
handler) cannot leave a stale reservation or lose real usage.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .errors import (
    RequestTooLarge,
    TenantRateLimited,
    UpstreamError,
    UpstreamRateLimited,
    UpstreamTimeout,
)
from .providers import (
    CompletionRequest,
    CompletionResult,
    Provider,
    ProviderError,
    ProviderRateLimited,
)
from .rate_limiter import (
    RateLimitExceeded,
    RequestExceedsLimit,
    SlidingWindowRateLimiter,
    estimate_tokens,
)

log = logging.getLogger("gateway.router")


@dataclass
class RouteOutcome:
    result: CompletionResult
    failed_over: bool
    failover_reason: str | None
    reserved_tokens: int
    actual_tokens: int


class ModelRouter:
    def __init__(
        self,
        primary: Provider,
        secondary: Provider,
        limiter: SlidingWindowRateLimiter,
        *,
        primary_timeout: float = 3.0,
        secondary_timeout: float = 10.0,
    ):
        self.primary = primary
        self.secondary = secondary
        self.limiter = limiter
        self.primary_timeout = primary_timeout
        self.secondary_timeout = secondary_timeout

    # -- public --------------------------------------------------------------------
    async def complete(self, tenant: str, req: CompletionRequest, *, request_id: str = "-") -> RouteOutcome:
        # Worst-case reservation: prompt estimate + everything the client allowed the model to emit.
        estimate = estimate_tokens(req.prompt_text()) + req.max_tokens
        try:
            reservation = await self.limiter.reserve(tenant, estimate)
        except RequestExceedsLimit as e:
            raise RequestTooLarge(detail=str(e)) from e
        except RateLimitExceeded as e:
            raise TenantRateLimited(
                f"Token budget for this API key is exhausted. Retry in {e.retry_after:.0f}s.",
                detail=str(e),
                retry_after=e.retry_after,
            ) from e

        try:
            result, reason = await self._call_with_failover(req, request_id)
        except BaseException:
            # Covers GatewayError, CancelledError (client went away), and bugs alike.
            await asyncio.shield(self.limiter.release(reservation))
            raise

        await asyncio.shield(self.limiter.commit(reservation, result.total_tokens))
        log.info(
            "request_id=%s tenant=%s provider=%s reserved=%d actual=%d failover=%s",
            request_id, tenant, result.provider, estimate, result.total_tokens, reason,
        )
        return RouteOutcome(result, reason is not None, reason, estimate, result.total_tokens)

    # -- internals -----------------------------------------------------------------
    async def _call_with_failover(self, req: CompletionRequest, request_id: str) -> tuple[CompletionResult, str | None]:
        try:
            return await self._call(self.primary, req, self.primary_timeout), None
        except ProviderRateLimited as e:
            reason = "primary_429"
            log.warning("request_id=%s primary=%s rate limited: %s", request_id, self.primary.name, e.detail)
        except TimeoutError:
            reason = "primary_timeout"
            log.warning("request_id=%s primary=%s timed out after %.0fms", request_id, self.primary.name, self.primary_timeout * 1000)
        except ProviderError as e:
            reason = f"primary_error_{e.status or 'conn'}"
            log.warning("request_id=%s primary=%s failed: %s", request_id, self.primary.name, e.detail)

        try:
            return await self._call(self.secondary, req, self.secondary_timeout), reason
        except ProviderRateLimited as e:
            log.error("request_id=%s secondary=%s rate limited: %s", request_id, self.secondary.name, e.detail)
            raise UpstreamRateLimited(detail=f"{reason}; secondary_429: {e.detail}") from None
        except TimeoutError:
            log.error("request_id=%s secondary=%s timed out", request_id, self.secondary.name)
            raise UpstreamTimeout(detail=f"{reason}; secondary_timeout") from None
        except ProviderError as e:
            log.error("request_id=%s secondary=%s failed: %s", request_id, self.secondary.name, e.detail)
            raise UpstreamError(detail=f"{reason}; secondary_error: {e.detail}") from None

    @staticmethod
    async def _call(provider: Provider, req: CompletionRequest, timeout: float) -> CompletionResult:
        try:
            async with asyncio.timeout(timeout):
                return await provider.complete(req)
        except (ProviderError, TimeoutError, asyncio.CancelledError):
            raise
        except Exception as e:
            # A buggy adapter must look like an upstream failure, not crash the request.
            raise ProviderError(provider.name, f"adapter raised {type(e).__name__}: {e}") from e
