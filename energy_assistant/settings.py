"""Local provider configuration. Keys are never returned to the browser."""

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


class ProviderSettings(BaseModel):
    provider: str = Field(pattern="^(guided|local|openai|anthropic|compatible|ollama)$")
    model: str = Field(default="", max_length=150)
    endpoint: str = Field(default="", max_length=500)
    api_key: str = Field(default="", max_length=1000)
    clear_key: bool = False


class SettingsStore:
    def __init__(self, directory, local_enabled=False, home=None, port=8771, initial=None):
        self.path = Path(directory) / (
            f"chat_settings_{home}.json" if home else "chat_settings.json"
        )
        self.lock = threading.Lock()
        self.local_enabled = local_enabled
        self.local_endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.value = {
            "provider": "local" if local_enabled else "guided",
            "model": "gridpfn-local",
            "endpoint": self.local_endpoint,
            "api_key": "",
        }
        if self.path.exists():
            self.value.update(json.loads(self.path.read_text()))
        elif os.environ.get("ENERGY_LLM_URL"):
            self.value.update(
                provider="compatible",
                endpoint=os.environ["ENERGY_LLM_URL"],
                model=os.environ.get("ENERGY_LLM_MODEL", ""),
                api_key=os.environ.get("ENERGY_LLM_KEY", ""),
            )
        if initial is not None:
            self.value = dict(
                initial
            )  # Explicit launch options override saved UI settings, in memory only.
        if self.value["provider"] == "local":
            self.value["endpoint"] = self.local_endpoint
            if not local_enabled:
                self.value["provider"] = "guided"

    def private(self):
        with self.lock:
            return dict(self.value)

    def public(self):
        value = self.private()
        key = value.pop("api_key", "")
        return value | {"has_key": bool(key), "local_available": self.local_enabled}

    @staticmethod
    def validate_endpoint(value):
        if value["provider"] in {"local", "guided"}:
            return
        parsed = urlsplit(value["endpoint"])
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an endpoint without credentials, query parameters or fragments")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("Use HTTPS for an external provider, or a localhost HTTP endpoint")
        if not parsed.hostname or not value["model"].strip():
            raise ValueError("Enter an endpoint and a model name")

    def save(self, update):
        value = update.model_dump()
        provider = value["provider"]
        if provider == "local":
            if not self.local_enabled:
                raise ValueError(
                    "Start the application with --llm qwen35_2b to use its built-in model"
                )
            value.update(
                model="gridpfn-local",
                endpoint=self.local_endpoint,
                api_key="",
            )
        elif provider == "openai":
            value["endpoint"] = "https://api.openai.com/v1/responses"
        elif provider == "anthropic":
            value["endpoint"] = "https://api.anthropic.com/v1/messages"
        self.validate_endpoint(value)
        with self.lock:
            if (
                not value["api_key"]
                and not value.pop("clear_key")
                and self.value["provider"] == provider
                and self.value["endpoint"] == value["endpoint"]
            ):
                value["api_key"] = self.value.get("api_key", "")
            value.pop("clear_key", None)
            if provider in {"local", "guided"}:
                value["api_key"] = ""
            temporary = self.path.with_suffix(".tmp")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle)
            temporary.replace(self.path)
            self.value = value
        return self.public()
