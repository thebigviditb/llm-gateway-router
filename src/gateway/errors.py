"""Gateway error taxonomy and the sanitized wire format.

Every error a client can observe is a GatewayError. The wire payload is always:

    {"error": {"code": "<STABLE_CODE>", "message": "<safe text>", "request_id": "<id>"}}

Upstream details (provider names, status bodies, stack traces) are logged server-side
under the request id and never placed in `message`.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("gateway.errors")


class GatewayError(Exception):
    """Base class. `message` is client-safe; `detail` is for logs only."""

    code = "INTERNAL_ERROR"
    status = 500
    default_message = "The gateway encountered an internal error."

    def __init__(self, message: str | None = None, *, detail: str | None = None,
                 retry_after: float | None = None):
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.detail = detail
        self.retry_after = retry_after


class InvalidRequest(GatewayError):
    code, status = "INVALID_REQUEST", 400
    default_message = "The request is malformed."


class Unauthorized(GatewayError):
    code, status = "UNAUTHORIZED", 401
    default_message = "A valid API key is required."


class TenantRateLimited(GatewayError):
    code, status = "TENANT_RATE_LIMITED", 429
    default_message = "Token budget for this API key is exhausted."


class RequestTooLarge(GatewayError):
    code, status = "REQUEST_TOO_LARGE", 413
    default_message = "The request exceeds the per-window token budget and can never be admitted."


class UpstreamRateLimited(GatewayError):
    code, status = "UPSTREAM_RATE_LIMITED", 503
    default_message = "All model providers are currently rate limited. Please retry."


class UpstreamTimeout(GatewayError):
    code, status = "UPSTREAM_TIMEOUT", 504
    default_message = "The model providers did not respond in time. Please retry."


class UpstreamError(GatewayError):
    code, status = "UPSTREAM_ERROR", 502
    default_message = "The model providers returned an error. Please retry."


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def error_response(request_id: str, err: GatewayError) -> JSONResponse:
    headers = {"X-Request-ID": request_id}
    if err.retry_after is not None:
        headers["Retry-After"] = str(max(1, int(err.retry_after + 0.999)))
    return JSONResponse(
        status_code=err.status,
        content={"error": {"code": err.code, "message": err.message, "request_id": request_id}},
        headers=headers,
    )


def install_handlers(app) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        rid = getattr(request.state, "request_id", None) or new_request_id()
        level = logging.WARNING if exc.status < 500 else logging.ERROR
        log.log(level, "request_id=%s code=%s detail=%s", rid, exc.code, exc.detail or exc.message)
        return error_response(rid, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None) or new_request_id()
        # Field locations are safe to surface; raw input echoes are not.
        fields = sorted({".".join(str(p) for p in e.get("loc", ()) if p != "body") for e in exc.errors()})
        msg = "Invalid request body." + (f" Problem fields: {', '.join(fields)}." if fields else "")
        return error_response(rid, InvalidRequest(msg))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None) or new_request_id()
        log.exception("request_id=%s unhandled exception", rid)
        return error_response(rid, GatewayError())
