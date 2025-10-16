"""Egress canary: try to open connections to external hosts from inside this process.

Used as a startup check in the API and dashboard (VIZOR_EGRESS_CANARY=1) and by
`vizor selfcheck egress`, to prove an offline run really has no network, not just no keys.
"""

from __future__ import annotations

import socket

TARGETS = [("1.1.1.1", 443), ("api.openai.com", 443), ("huggingface.co", 443)]


def check_egress(targets=TARGETS, timeout: float = 3.0) -> list[str]:
    """Return the targets that were reachable (empty list means egress is blocked)."""
    open_ = []
    for host, port in targets:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            open_.append(f"{host}:{port}")
        except OSError:
            pass
    return open_
