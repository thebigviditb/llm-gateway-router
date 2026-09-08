"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Model
-----
Each admitted request writes one ledger row (tenant, ts, tokens, state). The window
is a true sliding log: usage(tenant) = SUM(tokens) over rows with ts > now - window.

Two-phase accounting keeps the count accurate without knowing completion tokens up front:

    reserve(tenant, estimate) -> row in state 'reserved' with the estimate
    commit(rid, actual)       -> row updated to the provider's real usage, state 'committed'
    release(rid)              -> row deleted (the request never consumed tokens)

Concurrency
-----------
* A single aiosqlite connection is shared and every read-modify-write runs inside
  `BEGIN IMMEDIATE`, which takes SQLite's write lock up front. That makes
  "sum + estimate <= limit, then insert" atomic across coroutines *and* across
  processes sharing the same file.
* An asyncio.Lock additionally serializes coroutines on this connection so their
  statements cannot interleave inside one transaction.
* WAL mode + busy_timeout make cross-process contention wait instead of failing.

Eviction
--------
Rows older than the window are deleted at the start of every reserve() for that
tenant, so the table never grows unboundedly and the SUM is always over live rows.
A reservation that outlives the window (a stuck request) gets evicted too; commit()
handles that by re-inserting a committed row so the tokens are still counted.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Callable

import aiosqlite

Clock = Callable[[], float]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_ledger (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0),
    state   TEXT    NOT NULL CHECK (state IN ('reserved', 'committed'))
);
CREATE INDEX IF NOT EXISTS idx_ledger_tenant_ts ON token_ledger (tenant, ts);
"""


class RateLimitExceeded(Exception):
    def __init__(self, tenant: str, used: int, requested: int, limit: int, retry_after: float):
        super().__init__(f"tenant={tenant} used={used} requested={requested} limit={limit}")
        self.tenant, self.used, self.requested, self.limit = tenant, used, requested, limit
        self.retry_after = retry_after


class RequestExceedsLimit(Exception):
    """The estimate alone is larger than the whole budget; waiting will never help."""

    def __init__(self, requested: int, limit: int):
        super().__init__(f"requested={requested} limit={limit}")
        self.requested, self.limit = requested, limit


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int


class SlidingWindowRateLimiter:
    def __init__(self, db_path: str, *, limit: int, window_seconds: float, clock: Clock = time.time):
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self._db_path = db_path
        self.limit = int(limit)
        self.window = float(window_seconds)
        self._clock = clock
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------------
    async def open(self) -> "SlidingWindowRateLimiter":
        self._db = await aiosqlite.connect(self._db_path, isolation_level=None)  # autocommit; we BEGIN explicitly
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_SCHEMA)
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self):
        return await self.open()

    async def __aexit__(self, *exc):
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("rate limiter not opened; call open() or use `async with`")
        return self._db

    # -- internals -----------------------------------------------------------------
    async def _evict(self, tenant: str, now: float) -> None:
        await self.db.execute(
            "DELETE FROM token_ledger WHERE tenant = ? AND ts <= ?", (tenant, now - self.window)
        )

    async def _sum(self, tenant: str) -> int:
        async with self.db.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger WHERE tenant = ?", (tenant,)
        ) as cur:
            (used,) = await cur.fetchone()
        return int(used)

    async def _retry_after(self, tenant: str, now: float, need_to_free: int) -> float:
        """Seconds until enough of the oldest rows expire that `need_to_free` tokens are released."""
        freed = 0
        async with self.db.execute(
            "SELECT ts, tokens FROM token_ledger WHERE tenant = ? ORDER BY ts ASC", (tenant,)
        ) as cur:
            async for ts, tokens in cur:
                freed += tokens
                if freed >= need_to_free:
                    return max(0.0, ts + self.window - now)
        return self.window  # unreachable in practice; be conservative

    # -- public API ----------------------------------------------------------------
    async def reserve(self, tenant: str, estimate: int) -> Reservation:
        estimate = max(0, int(estimate))
        if estimate > self.limit:
            raise RequestExceedsLimit(estimate, self.limit)
        async with self._lock:
            now = self._clock()
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                await self._evict(tenant, now)
                used = await self._sum(tenant)
                if used + estimate > self.limit:
                    retry_after = await self._retry_after(tenant, now, used + estimate - self.limit)
                    await self.db.execute("ROLLBACK")
                    raise RateLimitExceeded(tenant, used, estimate, self.limit, retry_after)
                cur = await self.db.execute(
                    "INSERT INTO token_ledger (tenant, ts, tokens, state) VALUES (?, ?, ?, 'reserved')",
                    (tenant, now, estimate),
                )
                rid = cur.lastrowid
                await self.db.execute("COMMIT")
            except RateLimitExceeded:
                raise
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise
        return Reservation(id=rid, tenant=tenant, tokens=estimate)

    async def commit(self, reservation: Reservation, actual_tokens: int) -> None:
        """Reconcile the reservation to the provider-reported usage."""
        actual_tokens = max(0, int(actual_tokens))
        async with self._lock:
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                cur = await self.db.execute(
                    "UPDATE token_ledger SET tokens = ?, state = 'committed' WHERE id = ?",
                    (actual_tokens, reservation.id),
                )
                if cur.rowcount == 0:
                    # Reservation was evicted while the request was in flight (it outlived the
                    # window). The tokens were still consumed, so count them from now.
                    await self.db.execute(
                        "INSERT INTO token_ledger (tenant, ts, tokens, state) VALUES (?, ?, ?, 'committed')",
                        (reservation.tenant, self._clock(), actual_tokens),
                    )
                await self.db.execute("COMMIT")
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise

    async def release(self, reservation: Reservation) -> None:
        """Drop the reservation; the request consumed nothing."""
        async with self._lock:
            await self.db.execute("DELETE FROM token_ledger WHERE id = ?", (reservation.id,))

    async def usage(self, tenant: str) -> int:
        """Live tokens in the window (evicts first so the answer is exact)."""
        async with self._lock:
            now = self._clock()
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                await self._evict(tenant, now)
                used = await self._sum(tenant)
                await self.db.execute("COMMIT")
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise
        return used

    async def evict_all(self) -> int:
        """Global sweep for a background task; per-tenant eviction already happens on reserve."""
        async with self._lock:
            cur = await self.db.execute(
                "DELETE FROM token_ledger WHERE ts <= ?", (self._clock() - self.window,)
            )
            return cur.rowcount


def estimate_tokens(text: str) -> int:
    """Cheap pre-flight estimate (~4 chars/token). Always reconciled to real usage on commit."""
    return math.ceil(len(text) / 4)
