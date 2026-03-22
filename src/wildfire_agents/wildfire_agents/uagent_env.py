"""Shared env parsing for uAgent mailbox toggles."""

import os


def env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')
