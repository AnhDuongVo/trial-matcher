"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # the .env in the directory you run the command from

HOSTED_NIM_URL = "https://integrate.api.nvidia.com/v1"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("NIM_BASE_URL", HOSTED_NIM_URL))

    model_reasoning: str = field(
        default_factory=lambda: os.getenv("CTM_MODEL_REASONING", "nvidia/nemotron-3-super-120b-a12b")
    )
    model_fast: str = field(
        default_factory=lambda: os.getenv("CTM_MODEL_FAST", "nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    thinking: bool = field(default_factory=lambda: _bool("CTM_THINKING", False))
    temperature: float = field(default_factory=lambda: float(os.getenv("CTM_TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CTM_MAX_TOKENS", "4096")))

    def require_api_key(self) -> str:
        if not self.api_key and self.base_url.startswith(HOSTED_NIM_URL):
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Create a free key at https://build.nvidia.com and put it in .env"
            )
        return self.api_key or "not-needed-for-local-nim"


def get_settings() -> Settings:
    return Settings()
