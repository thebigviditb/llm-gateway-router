import asyncio

import pytest

from gateway.rate_limiter import RateLimitExceeded, RequestExceedsLimit, SlidingWindowRateLimiter


async def test_reserve_up_to_limit_then_reject(limiter):
    for _ in range(5):
        await limiter.reserve("t1", 10_000)
    assert await limiter.usage("t1") == 50_000
    with pytest.raises(RateLimitExceeded) as ei:
        await limiter.reserve("t1", 1)
    assert ei.value.used == 50_000
    assert ei.value.retry_after == pytest.approx(60.0)


async def test_tenants_are_isolated(limiter):
    await limiter.reserve("a", 50_000)
    await limiter.reserve("b", 50_000)  # unaffected by tenant a
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("a", 1)


async def test_sliding_window_evicts_exactly_at_boundary(limiter, clock):
    await limiter.reserve("t", 30_000)          # t=1000
    clock.advance(30)
    await limiter.reserve("t", 20_000)          # t=1030, full
    with pytest.raises(RateLimitExceeded) as ei:
        await limiter.reserve("t", 10_000)
    # Freeing 10k needs the 30k row to expire, at t=1060 -> 30s away.
    assert ei.value.retry_after == pytest.approx(30.0)

    clock.advance(29.999)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("t", 10_000)
    clock.advance(0.001)                        # t=1060: first row is now outside the window
    await limiter.reserve("t", 10_000)
    assert await limiter.usage("t") == 30_000   # 20k + 10k; the 30k row is gone


async def test_retry_after_accumulates_multiple_rows(limiter, clock):
    for i in range(5):
        await limiter.reserve("t", 10_000)
        clock.advance(5)                        # rows at 1000,1005,1010,1015,1020; now=1025
    with pytest.raises(RateLimitExceeded) as ei:
        await limiter.reserve("t", 25_000)      # needs 25k freed -> rows 1,2,3 expire at 1070
    assert ei.value.retry_after == pytest.approx(45.0)


async def test_commit_reconciles_to_actual_usage(limiter):
    res = await limiter.reserve("t", 5_000)
    assert await limiter.usage("t") == 5_000
    await limiter.commit(res, 1_234)
    assert await limiter.usage("t") == 1_234


async def test_release_returns_tokens(limiter):
    res = await limiter.reserve("t", 5_000)
    await limiter.release(res)
    assert await limiter.usage("t") == 0


async def test_commit_after_eviction_still_counts_tokens(limiter, clock):
    res = await limiter.reserve("t", 5_000)
    clock.advance(61)                           # reservation aged out while in flight
    assert await limiter.usage("t") == 0
    await limiter.commit(res, 900)
    assert await limiter.usage("t") == 900


async def test_estimate_larger_than_limit_is_rejected_outright(limiter):
    with pytest.raises(RequestExceedsLimit):
        await limiter.reserve("t", 50_001)


async def test_concurrent_burst_admits_exactly_the_budget(limiter):
    async def one():
        try:
            await limiter.reserve("t", 5_000)
            return True
        except RateLimitExceeded:
            return False

    results = await asyncio.gather(*(one() for _ in range(40)))
    assert sum(results) == 10                   # 10 * 5000 == 50000, not one more
    assert await limiter.usage("t") == 50_000


async def test_state_survives_reopen(tmp_path, clock):
    path = str(tmp_path / "persist.db")
    async with SlidingWindowRateLimiter(path, limit=50_000, window_seconds=60, clock=clock) as lim:
        await lim.reserve("t", 40_000)
    async with SlidingWindowRateLimiter(path, limit=50_000, window_seconds=60, clock=clock) as lim:
        assert await lim.usage("t") == 40_000
        with pytest.raises(RateLimitExceeded):
            await lim.reserve("t", 20_000)


async def test_two_connections_same_file_serialize(tmp_path, clock):
    """Cross-process safety proxy: two independent connections on one file."""
    path = str(tmp_path / "shared.db")
    async with SlidingWindowRateLimiter(path, limit=50_000, window_seconds=60, clock=clock) as a, \
               SlidingWindowRateLimiter(path, limit=50_000, window_seconds=60, clock=clock) as b:
        async def one(lim):
            try:
                await lim.reserve("t", 5_000)
                return True
            except RateLimitExceeded:
                return False

        results = await asyncio.gather(*(one(a if i % 2 else b) for i in range(30)))
        assert sum(results) == 10
        assert await a.usage("t") == 50_000
        assert await b.usage("t") == 50_000
