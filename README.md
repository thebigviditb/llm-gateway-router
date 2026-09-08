# llm-gateway-router

Resilient model-routing module for an LLM gateway: a **token-aware sliding-window rate
limiter** (SQLite on disk), **primary → secondary failover** on 429 / 3000 ms timeout, and
**sanitized gateway error payloads**.

```
client ──POST /v1/completions──▶ gateway
                                  │ 1. reserve(prompt_est + max_tokens)   ← SQLite ledger, BEGIN IMMEDIATE
                                  │ 2. primary  (asyncio.timeout 3.0s)
                                  │      429 / timeout / 5xx ──▶ secondary
                                  │ 3. commit(actual usage) | release() on failure
                                  ▼
                       {"text", "model", "failed_over", "usage"}   or   {"error": {code, message, request_id}}
```

## Quick start

```bash
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q

# Mock upstreams (no credentials). X-Mock-Primary / X-Mock-Secondary headers force ok|429|500|hang|crash.
GATEWAY_UPSTREAM=mock .venv/bin/uvicorn gateway.app:app --port 8080

curl -s localhost:8080/v1/completions -H 'X-API-Key: tenant-abc-123' -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'

# Force a primary 429 -> served by secondary, failed_over=true
curl -s ... -H 'X-Mock-Primary: 429' ...
# Both dead -> standardized 503/504, nothing upstream-specific in the body
curl -s ... -H 'X-Mock-Primary: hang' -H 'X-Mock-Secondary: 429' ...

# Real providers (Anthropic SDK; needs ANTHROPIC_API_KEY or `ant auth login`)
GATEWAY_UPSTREAM=anthropic GATEWAY_PRIMARY_MODEL=claude-opus-5 GATEWAY_SECONDARY_MODEL=claude-sonnet-5 \
  .venv/bin/uvicorn gateway.app:app --port 8080
```

## Layout

| File | Responsibility |
|---|---|
| `src/gateway/rate_limiter.py` | Sliding-window ledger in SQLite: `reserve` / `commit` / `release`, eviction, retry-after |
| `src/gateway/providers.py` | `Provider` protocol, `MockProvider` (scriptable), `AnthropicProvider` (official SDK, `max_retries=0`) |
| `src/gateway/router.py` | Failover state machine with the timeout race and token accounting |
| `src/gateway/errors.py` | Error taxonomy + handlers that emit the one wire shape and never leak upstream detail |
| `src/gateway/app.py` | FastAPI wiring, tenant identification, request ids |
| `tests/` | 33 tests: limiter accuracy/eviction/concurrency, failover mechanics, error sanitization |

## Design notes

### Rate limiter: two-phase token accounting
Completion tokens are unknown until the model answers, so a single "check-then-count" is
either inaccurate or unsafe. The ledger does:

1. **reserve** `estimate = ceil(prompt_chars/4) + max_tokens` (worst case) inside `BEGIN IMMEDIATE`.
   Eviction (`DELETE ts <= now - window`), `SUM`, compare, `INSERT` all happen under SQLite's
   write lock, so concurrent requests, including from other processes on the same file, cannot
   both squeeze through the last gap.
2. **commit** the provider's reported `input_tokens + output_tokens`, overwriting the estimate.
3. **release** on any failure (429 from both, timeout, client disconnect) so failed calls cost nothing.

`Retry-After` is exact: it walks the oldest rows and reports when enough of them fall out of
the window to admit the rejected request. A reservation that outlives the window is evicted
like any other row; `commit` re-inserts it so consumed tokens are never lost.

Commit/release run under `asyncio.shield`, so a client disconnect cancelling the handler
cannot strand a reservation.

### Failover and the timeout race
`asyncio.timeout(3.0)` wraps the primary call. When it fires, the primary coroutine is
cancelled *and the cancellation is awaited* before control returns, so a slow primary that
would have answered at 3.2 s can never reach the ledger. Only one provider result exists per
request. Triggers: primary 429, primary timeout, and (extension) any other upstream/adapter
failure. The secondary gets its own, longer timeout. If it also fails the client gets
`UPSTREAM_RATE_LIMITED` (503), `UPSTREAM_TIMEOUT` (504) or `UPSTREAM_ERROR` (502).

The Anthropic adapter sets `max_retries=0` so the SDK's built-in 429 retry cannot silently
absorb the failover signal or stretch a call past the gateway deadline.

### Error sanitization
Every non-2xx body is `{"error": {"code", "message", "request_id"}}` plus an `X-Request-ID`
header (and `Retry-After` where meaningful). Handlers exist for `GatewayError`, pydantic
validation errors (only field *names* are echoed, never values) and a catch-all `Exception`.
Upstream messages, provider names, status bodies and tracebacks are logged under the request
id and never serialized. Tenants are identified by a SHA-256 digest of the API key; raw keys
are never written to disk.

### Known trade-offs
- Reserving `max_tokens` is conservative: a tenant asking for large `max_tokens` can be
  throttled on estimates even if real usage is small. The alternative (reserve prompt only,
  allow overshoot) trades exactness for throughput; the requirement here favoured exactness.
- The mock-behaviour headers mutate shared provider state and are for demos only; they are
  ignored unless `GATEWAY_UPSTREAM=mock`.
- No streaming. Streaming failover after first byte is a different problem (you cannot switch
  providers mid-response) and is out of scope.

## Requirement traceability

A requirement-by-requirement map of the implementation (mechanism, code lines, tests, likely questions) is in [docs/traceability.html](docs/traceability.html). Open it in a browser.
