"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml


_config_cache: dict[str, Any] = {}

# UI session overrides — set by Streamlit app; always empty for CLI.
_ui_overrides: dict[str, Any] = {}


def set_ui_overrides(overrides: dict) -> None:
    """Replace the UI override store. Called by Streamlit on every render."""
    global _ui_overrides
    _ui_overrides = overrides


def clear_ui_overrides() -> None:
    """Clear all UI overrides (reverts to settings.yaml)."""
    global _ui_overrides
    _ui_overrides = {}


def _find_config_dir() -> Path:
    """Find the config directory relative to project root."""
    current = Path(__file__).resolve().parent
    # Walk up to find config/ directory
    for _ in range(5):
        config_dir = current / "config"
        if config_dir.exists():
            return config_dir
        current = current.parent
    raise FileNotFoundError("Could not find config/ directory")


def load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load settings.yaml and return as dict."""
    if "settings" in _config_cache:
        return _config_cache["settings"]

    if config_path is None:
        config_path = _find_config_dir() / "settings.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        settings = yaml.safe_load(f)

    _config_cache["settings"] = settings
    return settings


def load_nifty50_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load nifty50.yaml and return as dict."""
    if "nifty50" in _config_cache:
        return _config_cache["nifty50"]

    if config_path is None:
        config_path = _find_config_dir() / "nifty50.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _config_cache["nifty50"] = config
    return config


def get_setting(*keys: str, default: Any = None) -> Any:
    """Get a nested setting value. Example: get_setting('models', 'lstm', 'epochs')"""
    # Check UI session overrides first (Streamlit only; always empty in CLI)
    if _ui_overrides:
        value = _ui_overrides
        found = True
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break
        if found:
            return value
    # Fall back to settings.yaml
    settings = load_settings()
    value = settings
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def clear_config_cache() -> None:
    """Clear cached configurations."""
    _config_cache.clear()
