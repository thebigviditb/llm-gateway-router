from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from pydantic import BaseModel

from .config import Settings
from .errors import InvalidRequest, Unauthorized, install_handlers, new_request_id
from .providers import AnthropicProvider, CompletionRequest, MockProvider
from .rate_limiter import SlidingWindowRateLimiter
from .router import ModelRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("gateway.app")


def build_providers(settings: Settings) -> tuple:
    if settings.upstream == "anthropic":
        return (
            AnthropicProvider("primary", settings.primary_model),
            AnthropicProvider("secondary", settings.secondary_model),
        )
    return MockProvider(name="primary", model="mock-primary"), MockProvider(name="secondary", model="mock-secondary")


def create_app(settings: Settings | None = None, *, primary=None, secondary=None, limiter=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lim = limiter or SlidingWindowRateLimiter(
            settings.db_path, limit=settings.token_limit, window_seconds=settings.window_seconds
        )
        if limiter is None:
            await lim.open()
        p, s = (primary, secondary) if primary and secondary else build_providers(settings)
        app.state.primary, app.state.secondary = p, s
        app.state.router = ModelRouter(
            p, s, lim,
            primary_timeout=settings.primary_timeout_seconds,
            secondary_timeout=settings.secondary_timeout_seconds,
        )
        log.info("gateway up: upstream=%s db=%s limit=%d/%ss", settings.upstream, settings.db_path,
                 settings.token_limit, settings.window_seconds)
        try:
            yield
        finally:
            if limiter is None:
                await lim.close()

    app = FastAPI(title="LLM Gateway Router", lifespan=lifespan)
    app.state.settings = settings
    install_handlers(app)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or new_request_id()
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        return response

    def tenant_id(x_api_key: str | None = Header(default=None)) -> str:
        if not x_api_key or len(x_api_key) < 8:
            raise Unauthorized()
        # Never persist the raw key; the ledger only sees a stable digest.
        return hashlib.sha256(x_api_key.encode()).hexdigest()[:24]

    class CompletionResponse(BaseModel):
        request_id: str
        text: str
        model: str
        failed_over: bool
        usage: dict

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(
        request: Request,
        body: CompletionRequest,
        tenant: str = Depends(tenant_id),
        x_mock_primary: str | None = Header(default=None),
        x_mock_secondary: str | None = Header(default=None),
    ):
        if body.max_tokens > settings.max_tokens_cap:
            raise InvalidRequest(f"max_tokens may not exceed {settings.max_tokens_cap}.")
        # Demo hook: only honored when running against mock providers.
        if settings.upstream == "mock":
            _apply_mock_behavior(app.state.primary, x_mock_primary)
            _apply_mock_behavior(app.state.secondary, x_mock_secondary)

        rid = request.state.request_id
        outcome = await app.state.router.complete(tenant, body, request_id=rid)
        r = outcome.result
        return CompletionResponse(
            request_id=rid,
            text=r.text,
            model=r.model,
            failed_over=outcome.failed_over,
            usage={
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "reserved_tokens": outcome.reserved_tokens,
            },
        )

    @app.get("/v1/usage")
    async def usage(tenant: str = Depends(tenant_id)):
        used = await app.state.router.limiter.usage(tenant)
        return {"tokens_used": used, "limit": settings.token_limit, "window_seconds": settings.window_seconds}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


def _apply_mock_behavior(provider, behavior: str | None) -> None:
    if behavior and isinstance(provider, MockProvider) and behavior in ("ok", "429", "500", "hang", "crash"):
        provider.behavior = behavior


app = create_app()
