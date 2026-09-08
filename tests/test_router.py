import asyncio
import time

import pytest

from gateway.errors import TenantRateLimited, UpstreamError, UpstreamRateLimited, UpstreamTimeout
from gateway.providers import CompletionRequest, Message, MockProvider
from gateway.rate_limiter import estimate_tokens
from gateway.router import ModelRouter

REQ = CompletionRequest(messages=[Message(role="user", content="hello world")], max_tokens=100)


def make(limiter, primary_behavior="ok", secondary_behavior="ok", **kw):
    p = MockProvider(name="primary", model="p-model", behavior=primary_behavior, output_tokens=40)
    s = MockProvider(name="secondary", model="s-model", behavior=secondary_behavior, output_tokens=70)
    kw.setdefault("primary_timeout", 0.05)
    kw.setdefault("secondary_timeout", 0.2)
    return ModelRouter(p, s, limiter, **kw), p, s


async def test_primary_success_no_failover_and_tokens_reconciled(limiter):
    router, p, s = make(limiter)
    out = await router.complete("t", REQ)
    assert out.result.provider == "primary" and not out.failed_over
    assert s.calls == 0
    expected_reserved = estimate_tokens(REQ.prompt_text()) + REQ.max_tokens
    assert out.reserved_tokens == expected_reserved
    assert out.actual_tokens == out.result.total_tokens < expected_reserved
    assert await limiter.usage("t") == out.actual_tokens


async def test_primary_429_fails_over(limiter):
    router, p, s = make(limiter, primary_behavior="429")
    out = await router.complete("t", REQ)
    assert out.failed_over and out.failover_reason == "primary_429"
    assert out.result.provider == "secondary"
    assert await limiter.usage("t") == out.result.total_tokens  # secondary's usage, not primary's


async def test_primary_timeout_fails_over_and_cancels_primary(limiter):
    router, p, s = make(limiter, primary_behavior="hang", primary_timeout=0.05)
    t0 = time.monotonic()
    out = await router.complete("t", REQ)
    elapsed = time.monotonic() - t0
    assert out.failover_reason == "primary_timeout"
    assert out.result.provider == "secondary"
    assert elapsed < 0.5                      # did not wait for the hung primary
    assert p.cancelled == 1 and p.completed == 0


async def test_slow_primary_result_after_deadline_is_discarded(limiter):
    """Primary would succeed at 0.1s but deadline is 0.05s: only secondary's tokens land."""
    router, p, s = make(limiter, primary_timeout=0.05)
    p.delay = 0.1
    out = await router.complete("t", REQ)
    await asyncio.sleep(0.15)                 # give a leaked primary time to misbehave, if it could
    assert out.result.provider == "secondary"
    assert p.completed == 0 and p.cancelled == 1
    assert await limiter.usage("t") == s.output_tokens + out.result.input_tokens


async def test_primary_within_deadline_wins(limiter):
    router, p, s = make(limiter, primary_timeout=0.2)
    p.delay = 0.02
    out = await router.complete("t", REQ)
    assert out.result.provider == "primary" and s.calls == 0


async def test_primary_5xx_fails_over(limiter):
    router, p, s = make(limiter, primary_behavior="500")
    out = await router.complete("t", REQ)
    assert out.failover_reason == "primary_error_500" and out.result.provider == "secondary"


async def test_adapter_crash_is_treated_as_upstream_failure(limiter):
    router, p, s = make(limiter, primary_behavior="crash")
    out = await router.complete("t", REQ)
    assert out.result.provider == "secondary"


@pytest.mark.parametrize(
    "p_beh,s_beh,exc",
    [("429", "429", UpstreamRateLimited), ("hang", "hang", UpstreamTimeout), ("429", "500", UpstreamError)],
)
async def test_both_fail_raises_sanitized_error_and_releases_reservation(limiter, p_beh, s_beh, exc):
    router, p, s = make(limiter, p_beh, s_beh, primary_timeout=0.05, secondary_timeout=0.05)
    with pytest.raises(exc) as ei:
        await router.complete("t", REQ)
    assert "DEADBEEF" not in ei.value.message and "stack" not in ei.value.message
    assert await limiter.usage("t") == 0      # nothing consumed, reservation released


async def test_tenant_limit_blocks_before_any_provider_call(limiter):
    router, p, s = make(limiter)
    big = CompletionRequest(messages=[Message(role="user", content="x")], max_tokens=49_000)
    await router.complete("t", big)           # reserves ~49001, commits 1+40
    await limiter.reserve("t", 49_000)        # fill the window
    with pytest.raises(TenantRateLimited) as ei:
        await router.complete("t", big)
    assert ei.value.retry_after is not None and ei.value.status == 429
    assert p.calls == 1 and s.calls == 0


async def test_client_disconnect_releases_reservation(limiter):
    router, p, s = make(limiter, primary_timeout=5)
    p.delay = 1
    task = asyncio.create_task(router.complete("t", REQ))
    await asyncio.sleep(0.02)
    assert await limiter.usage("t") > 0       # reserved while in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await limiter.usage("t") == 0


async def test_concurrent_requests_respect_budget(limiter):
    router, p, s = make(limiter)
    req = CompletionRequest(messages=[Message(role="user", content="x")], max_tokens=10_000)
    outs = await asyncio.gather(*(router.complete("t", req) for _ in range(20)), return_exceptions=True)
    ok = [o for o in outs if not isinstance(o, Exception)]
    limited = [o for o in outs if isinstance(o, TenantRateLimited)]
    # Reservations are worst-case (10001 each) so at most 4 fit at once even though actual usage is tiny.
    assert len(ok) >= 4 and len(ok) + len(limited) == 20
    assert await limiter.usage("t") == sum(o.actual_tokens for o in ok)
