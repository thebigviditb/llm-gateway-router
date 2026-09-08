import pytest

from gateway.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
async def limiter(tmp_path, clock):
    async with SlidingWindowRateLimiter(
        str(tmp_path / "ledger.db"), limit=50_000, window_seconds=60, clock=clock
    ) as lim:
        yield lim
