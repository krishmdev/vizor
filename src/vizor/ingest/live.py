"""Live fetch of individual pages: checks robots.txt, identifies itself, rate-limits per host."""

from __future__ import annotations

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from vizor.ingest.commoncrawl import USER_AGENT
from vizor.ingest.extract import extract
from vizor.types import SourceDoc

_last_hit: dict[str, float] = {}
_robots: dict[str, RobotFileParser] = {}


class Disallowed(RuntimeError):
    pass


def allowed(url: str, client: httpx.Client) -> bool:
    host = urlparse(url)
    root = f"{host.scheme}://{host.netloc}"
    if root not in _robots:
        rp = RobotFileParser()
        try:
            r = client.get(root + "/robots.txt")
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except httpx.HTTPError:
            rp.parse([])
        _robots[root] = rp
    return _robots[root].can_fetch(USER_AGENT, url)


def fetch_page(
    url: str,
    role: str = "competitor",
    min_interval_s: float = 2.0,
    client: httpx.Client | None = None,
) -> SourceDoc:
    client = client or httpx.Client(
        timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )
    if not allowed(url, client):
        raise Disallowed(f"robots.txt disallows {url}")
    host = urlparse(url).netloc
    wait = min_interval_s - (time.monotonic() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    r = client.get(url)
    _last_hit[host] = time.monotonic()
    r.raise_for_status()
    prov = {"source": "live", "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return extract(r.text, str(r.url), role=role, provenance=prov)  # type: ignore[arg-type]
