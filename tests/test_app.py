import pytest
from httpx import ASGITransport, AsyncClient

from gateway.app import create_app
from gateway.config import Settings
from gateway.providers import MockProvider
from gateway.rate_limiter import SlidingWindowRateLimiter

KEY = {"X-API-Key": "tenant-key-12345"}
BODY = {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 64}


@pytest.fixture
async def client(tmp_path):
    settings = Settings(upstream="mock", primary_timeout_seconds=0.05, secondary_timeout_seconds=0.05,
                        token_limit=50_000, window_seconds=60)
    lim = await SlidingWindowRateLimiter(str(tmp_path / "app.db"), limit=50_000, window_seconds=60).open()
    p, s = MockProvider(name="primary"), MockProvider(name="secondary")
    app = create_app(settings, primary=p, secondary=s, limiter=lim)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw") as c:
            c.mocks = (p, s)
            yield c
    await lim.close()


def assert_error_shape(resp, code, status):
    assert resp.status_code == status
    body = resp.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == resp.headers["X-Request-ID"]
    txt = resp.text.lower()
    for leak in ("traceback", "deadbeef", "stack", "exception", "line 42", "mock", "sqlite"):
        assert leak not in txt, f"leaked '{leak}' in error body"


async def test_success_shape(client):
    r = await client.post("/v1/completions", json=BODY, headers=KEY)
    assert r.status_code == 200
    j = r.json()
    assert j["failed_over"] is False and j["usage"]["reserved_tokens"] > j["usage"]["output_tokens"]
    assert "X-Request-ID" in r.headers


async def test_missing_api_key(client):
    assert_error_shape(await client.post("/v1/completions", json=BODY), "UNAUTHORIZED", 401)


async def test_validation_error_is_standardized(client):
    r = await client.post("/v1/completions", json={"messages": []}, headers=KEY)
    assert_error_shape(r, "INVALID_REQUEST", 400)
    assert "messages" in r.json()["error"]["message"]


async def test_failover_via_mock_header(client):
    r = await client.post("/v1/completions", json=BODY, headers={**KEY, "X-Mock-Primary": "429"})
    assert r.status_code == 200 and r.json()["failed_over"] is True


async def test_both_upstreams_429_sanitized(client):
    r = await client.post("/v1/completions", json=BODY,
                          headers={**KEY, "X-Mock-Primary": "429", "X-Mock-Secondary": "429"})
    assert_error_shape(r, "UPSTREAM_RATE_LIMITED", 503)


async def test_both_upstreams_timeout_sanitized(client):
    r = await client.post("/v1/completions", json=BODY,
                          headers={**KEY, "X-Mock-Primary": "hang", "X-Mock-Secondary": "hang"})
    assert_error_shape(r, "UPSTREAM_TIMEOUT", 504)


async def test_failed_requests_consume_no_budget(client):
    big = {**BODY, "max_tokens": 8000}
    for _ in range(3):
        assert (await client.post("/v1/completions", json=big, headers=KEY)).status_code == 200
    before = (await client.get("/v1/usage", headers=KEY)).json()["tokens_used"]
    r = await client.post("/v1/completions", json=big,
                          headers={**KEY, "X-Mock-Primary": "hang", "X-Mock-Secondary": "hang"})
    assert r.status_code == 504
    after = (await client.get("/v1/usage", headers=KEY)).json()["tokens_used"]
    assert after == before                    # 8000-token reservation was released, not committed
    assert before < 3 * 8000                  # and successes were reconciled down to real usage


async def test_tenant_rate_limit_triggers(tmp_path):
    settings = Settings(upstream="mock", token_limit=300, window_seconds=60)
    lim = await SlidingWindowRateLimiter(str(tmp_path / "small.db"), limit=300, window_seconds=60).open()
    app = create_app(settings, primary=MockProvider(name="p", output_tokens=100),
                     secondary=MockProvider(name="s"), limiter=lim)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw") as c:
            body = {**BODY, "max_tokens": 100}
            assert (await c.post("/v1/completions", json=body, headers=KEY)).status_code == 200
            assert (await c.post("/v1/completions", json=body, headers=KEY)).status_code == 200
            r = await c.post("/v1/completions", json=body, headers=KEY)
            assert_error_shape(r, "TENANT_RATE_LIMITED", 429)
            assert int(r.headers["Retry-After"]) >= 1
            r = await c.post("/v1/completions", json={**BODY, "max_tokens": 301}, headers=KEY)
            assert_error_shape(r, "REQUEST_TOO_LARGE", 413)
    await lim.close()


async def test_unhandled_exception_is_sanitized(client):
    app = client._transport.app
    app.state.router.complete = None  # force a TypeError inside the handler
    r = await client.post("/v1/completions", json=BODY, headers=KEY)
    assert_error_shape(r, "INTERNAL_ERROR", 500)
