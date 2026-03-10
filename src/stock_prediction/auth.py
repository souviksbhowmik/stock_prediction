"""User profile management for multi-user Streamlit UI.

The CLI is unaffected — it always uses the default profile (data/models/).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stock_prediction.config import get_setting

_REGISTRY_PATH = Path("data/profiles/users.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UserProfile:
    username: str
    display_name: str
    role: str = "user"          # "user" | "admin" | "default"
    created_at: str = field(default_factory=_now_iso)

    @property
    def save_dir(self) -> Path:
        if self.username == "default":
            return Path(get_setting("models", "save_dir", default="data/models"))
        return Path("data/profiles") / self.username / "models"

    @property
    def plots_dir(self) -> Path:
        return self.save_dir.parent / "plots"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict[str, dict]:
    if _REGISTRY_PATH.exists():
        try:
            with open(_REGISTRY_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(registry: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _ensure_builtin_users(registry: dict[str, dict]) -> bool:
    """Ensure default and admin users exist. Returns True if registry was modified."""
    changed = False
    if "default" not in registry:
        registry["default"] = {
            "username": "default",
            "display_name": "Default",
            "role": "default",
            "created_at": _now_iso(),
        }
        changed = True
    if "admin" not in registry:
        registry["admin"] = {
            "username": "admin",
            "display_name": "Admin",
            "role": "admin",
            "created_at": _now_iso(),
        }
        changed = True
    return changed


def _load_and_ensure() -> dict[str, dict]:
    registry = _load_registry()
    if _ensure_builtin_users(registry):
        _save_registry(registry)
    return registry


def _profile_from_dict(d: dict) -> UserProfile:
    return UserProfile(
        username=d["username"],
        display_name=d.get("display_name", d["username"]),
        role=d.get("role", "user"),
        created_at=d.get("created_at", _now_iso()),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def authenticate(username: str, password: str) -> UserProfile:  # noqa: ARG001
    """Accept any password. Auto-registers unknown usernames."""
    username = username.strip().lower()
    if not username:
        raise ValueError("Username cannot be empty.")

    registry = _load_and_ensure()
    if username not in registry:
        # Auto-register as regular user
        registry[username] = {
            "username": username,
            "display_name": username.capitalize(),
            "role": "user",
            "created_at": _now_iso(),
        }
        _save_registry(registry)

    return _profile_from_dict(registry[username])


def get_profile(username: str) -> Optional[UserProfile]:
    registry = _load_and_ensure()
    d = registry.get(username.strip().lower())
    return _profile_from_dict(d) if d else None


def list_users() -> list[UserProfile]:
    registry = _load_and_ensure()
    return [_profile_from_dict(d) for d in registry.values()]


def delete_user(username: str) -> None:
    """Remove user from registry and delete their data directory.

    Raises ValueError for 'default' and 'admin'.
    """
    username = username.strip().lower()
    if username in ("default", "admin"):
        raise ValueError(f"Cannot delete built-in user '{username}'.")

    registry = _load_and_ensure()
    if username not in registry:
        raise ValueError(f"User '{username}' not found.")

    del registry[username]
    _save_registry(registry)

    user_dir = Path("data/profiles") / username
    if user_dir.exists():
        shutil.rmtree(user_dir)


def clear_user_data(username: str) -> tuple[int, int]:
    """Delete model and plots directories but keep the account.

    Returns (files_deleted, bytes_freed).
    """
    profile = get_profile(username)
    if profile is None:
        raise ValueError(f"User '{username}' not found.")

    files_deleted = 0
    bytes_freed = 0

    for target_dir in (profile.save_dir, profile.plots_dir):
        if target_dir.exists():
            for f in target_dir.rglob("*"):
                if f.is_file():
                    bytes_freed += f.stat().st_size
                    files_deleted += 1
            shutil.rmtree(target_dir)

    return files_deleted, bytes_freed
