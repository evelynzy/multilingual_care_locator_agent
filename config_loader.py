from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


class ConfigError(RuntimeError):
    """Raised when required configuration entries are missing."""


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise ConfigError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def get_prompt(name: str, default: Optional[str] = "") -> str:
    prompts = _load_config().get("prompts", {})
    return prompts.get(name, default or "")


def get_chat_model_settings() -> Dict[str, Any]:
    models = _load_config().get("models", {})
    chat = models.get("chat", {})
    env_key = chat.get("env")
    model_id = os.getenv(env_key, chat.get("default_id")) if env_key else chat.get("default_id")
    return {
        "model_id": model_id,
        "max_tokens": chat.get("max_tokens", 512),
        "temperature": chat.get("temperature", 0.7),
        "top_p": chat.get("top_p", 0.9),
    }


def get_embed_model_name() -> str:
    models = _load_config().get("models", {})
    embed = models.get("embed", {})
    env_key = embed.get("env")
    return os.getenv(env_key, embed.get("default_id")) if env_key else embed.get("default_id")


def get_search_settings() -> Dict[str, Any]:
    search = _load_config().get("search", {})
    clinical = search.get("clinicaltables", {})
    registry = search.get("npi_registry", {})
    return {
        "default_top_k": search.get("default_top_k", 5),
        "clinicaltables": {
            "individual_search_url": clinical.get(
                "individual_search_url",
                "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/search",
            ),
            "individual_values_url": clinical.get(
                "individual_values_url",
                "https://clinicaltables.nlm.nih.gov/api/npi_idv/v3/values",
            ),
            "organization_search_url": clinical.get(
                "organization_search_url",
                "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/search",
            ),
            "organization_values_url": clinical.get(
                "organization_values_url",
                "https://clinicaltables.nlm.nih.gov/api/npi_org/v3/values",
            ),
            "field_probe_terms": clinical.get("field_probe_terms", {}),
            "timeout": clinical.get("timeout", 6),
            "max_results": clinical.get("max_results", 3),
            "values_max_results": clinical.get("values_max_results", 10),
        },
        "npi_registry": {
            "lookup_url": registry.get(
                "lookup_url", "https://npiregistry.cms.hhs.gov/api/"
            ),
            "version": str(registry.get("version", "2.1")),
            "timeout": registry.get("timeout", 6),
            "enabled": registry.get("enabled", True),
        },
        "fallback_resources": search.get("fallback_resources", []),
    }


def get_ui_settings() -> Dict[str, Any]:
    ui = _load_config().get("ui", {})
    slider_defaults = ui.get("sliders", {})
    return {
        "title": ui.get("chat_title", "Multilingual Care Locator"),
        "description": ui.get("chat_description", ""),
        "sliders": slider_defaults,
        "default_system_message": get_prompt("default_system_message"),
    }


def get_message(key: str, default: str = "") -> str:
    messages = _load_config().get("messages", {})
    return messages.get(key, default)


__all__ = [
    "ConfigError",
    "get_prompt",
    "get_chat_model_settings",
    "get_embed_model_name",
    "get_search_settings",
    "get_ui_settings",
    "get_message",
]
