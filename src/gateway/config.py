from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    upstream: str = "mock"                  # "mock" | "anthropic"
    db_path: str = "gateway.db"
    token_limit: int = 50_000               # tokens per window per tenant
    window_seconds: float = 60.0
    primary_timeout_seconds: float = 3.0    # spec: 3000ms
    secondary_timeout_seconds: float = 10.0
    primary_model: str = "claude-opus-5"
    secondary_model: str = "claude-sonnet-5"
    max_tokens_cap: int = 8_192             # hard cap on max_tokens a client may request

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            upstream=env("GATEWAY_UPSTREAM", cls.upstream),
            db_path=env("GATEWAY_DB_PATH", cls.db_path),
            token_limit=int(env("GATEWAY_TOKEN_LIMIT", cls.token_limit)),
            window_seconds=float(env("GATEWAY_WINDOW_SECONDS", cls.window_seconds)),
            primary_timeout_seconds=float(env("GATEWAY_PRIMARY_TIMEOUT_MS", cls.primary_timeout_seconds * 1000)) / 1000,
            secondary_timeout_seconds=float(env("GATEWAY_SECONDARY_TIMEOUT_MS", cls.secondary_timeout_seconds * 1000)) / 1000,
            primary_model=env("GATEWAY_PRIMARY_MODEL", cls.primary_model),
            secondary_model=env("GATEWAY_SECONDARY_MODEL", cls.secondary_model),
            max_tokens_cap=int(env("GATEWAY_MAX_TOKENS_CAP", cls.max_tokens_cap)),
        )
