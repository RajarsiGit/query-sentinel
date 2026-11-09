"""Runtime configuration, read from environment variables with sane local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    dsn: str = field(default_factory=lambda: _env("SENTINEL_DSN", "postgresql://sentinel:sentinel@localhost:5432/shop"))
    snapshot_interval_s: int = field(default_factory=lambda: int(_env("SNAPSHOT_INTERVAL", "30")))
    min_calls: int = field(default_factory=lambda: int(_env("MIN_CALLS", "5")))
    baseline_windows: int = field(default_factory=lambda: int(_env("BASELINE_WINDOWS", "5")))
    horizon: int = field(default_factory=lambda: int(_env("HORIZON", "3")))
    regression_ratio: float = field(default_factory=lambda: float(_env("REGRESSION_RATIO", "1.5")))
    alert_threshold: float = field(default_factory=lambda: float(_env("ALERT_THRESHOLD", "0.7")))
    bucket: str = field(default_factory=lambda: _env("SENTINEL_BUCKET", ""))
    endpoint_name: str = field(default_factory=lambda: _env("SENTINEL_ENDPOINT", "query-sentinel"))
    sns_topic_arn: str = field(default_factory=lambda: _env("SENTINEL_TOPIC_ARN", ""))
    region: str = field(default_factory=lambda: _env("AWS_REGION", "ap-south-1"))


settings = Settings()
