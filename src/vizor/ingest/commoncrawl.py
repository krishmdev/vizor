"""Common Crawl: CDX index lookup -> WARC byte-range fetch -> extract.

Ported from the old app/ingestion/commoncrawl.py (search_articles, fetch_article) with these fixes:
the CDX lines are parsed with json.loads (the old code ran eval() on remote data), WARC records
are parsed with warcio instead of by hand, `index="latest"` resolves through collinfo.json,
requests back off on 503/SlowDown at about one request per second, and fetched WARC slices are
cached under .cache/warc/.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from vizor.ingest.extract import extract
from vizor.types import SourceDoc

INDEX_ROOT = "https://index.commoncrawl.org"
DATA_ROOT = "https://data.commoncrawl.org"
MAX_RECORD_BYTES = 5_000_000
USER_AGENT = "vizor/0.2 (research; +https://github.com/krishmdev/vizor)"


def _throttled(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


@dataclass(frozen=True)
class CdxRecord:
    url: str
    filename: str
    offset: int
    length: int
    status: str
    mime: str
    timestamp: str

    @classmethod
    def from_json(cls, line: str) -> CdxRecord:
        d = json.loads(line)
        return cls(
            url=d["url"],
            filename=d["filename"],
            offset=int(d["offset"]),
            length=int(d["length"]),
            status=str(d.get("status", "")),
            mime=d.get("mime", d.get("mime-detected", "")),
            timestamp=d.get("timestamp", ""),
        )


def parse_cdx(text: str) -> list[CdxRecord]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CdxRecord.from_json(line))
        except (ValueError, KeyError):
            continue
    return out


def parse_warc(data: bytes) -> tuple[str, str, int] | None:
    """(target URI, HTML, HTTP status) from the first response record of a WARC slice."""
    from warcio.archiveiterator import ArchiveIterator

    for rec in ArchiveIterator(io.BytesIO(data)):
        if rec.rec_type != "response":
            continue
        uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
        status = int(rec.http_headers.get_statuscode() or 0) if rec.http_headers else 0
        body = rec.content_stream().read()
        ctype = rec.http_headers.get_header("Content-Type") if rec.http_headers else ""
        charset = "utf-8"
        if ctype and "charset=" in ctype:
            charset = ctype.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            html = body.decode(charset, errors="replace")
        except LookupError:
            html = body.decode("utf-8", errors="replace")
        return uri, html, status
    return None


class CommonCrawl:
    def __init__(
        self,
        index: str = "latest",
        cache_dir: Path | None = None,
        min_interval_s: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        from vizor.config import cache_dir as default_cache

        self.http = client or httpx.Client(
            timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
        self.cache = (cache_dir or default_cache()) / "warc"
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.errors: list[str] = []
        self.index_name = self.resolve_index(index)

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.monotonic()

    @retry(
        retry=retry_if_exception(_throttled),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _get(self, url: str, **kw) -> httpx.Response:
        self._wait()
        r = self.http.get(url, **kw)
        r.raise_for_status()
        return r

    def _range(self, url: str, headers: dict, length: int) -> bytes:
        """Stream a byte range. Anything but 206 Partial Content (e.g. a server ignoring Range and
        sending the whole 1 GB WARC file) is refused, and reading stops at the requested length."""
        self._wait()
        with self.http.stream("GET", url, headers=headers) as r:
            if r.status_code in (429, 500, 502, 503, 504):
                r.raise_for_status()
            if r.status_code != 206:
                raise ValueError(f"expected 206 for a range request, got {r.status_code}")
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > length:
                    raise ValueError("server sent more bytes than the requested range")
            return bytes(buf)

    def resolve_index(self, index: str) -> str:
        if index != "latest":
            return index
        return self._get(f"{INDEX_ROOT}/collinfo.json").json()[0]["id"]

    def search(self, pattern: str, limit: int = 10) -> list[CdxRecord]:
        params = {
            "url": pattern,
            "output": "json",
            "limit": str(limit),
            "filter": ["status:200", "mime:text/html"],
            "fl": "url,filename,offset,length,status,mime,timestamp",
        }
        try:
            r = self._get(f"{INDEX_ROOT}/{self.index_name}-index", params=params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:  # no captures for this pattern
                return []
            raise
        return parse_cdx(r.text)[:limit]

    def fetch(self, rec: CdxRecord) -> bytes:
        key = hashlib.sha1(f"{rec.filename}:{rec.offset}:{rec.length}".encode()).hexdigest()
        path = self.cache / f"{key}.warc.gz"
        if path.exists():
            return path.read_bytes()
        if rec.length > MAX_RECORD_BYTES:
            raise ValueError(f"record of {rec.length} bytes is over the {MAX_RECORD_BYTES} cap")
        headers = {"Range": f"bytes={rec.offset}-{rec.offset + rec.length - 1}"}
        data = self._range(f"{DATA_ROOT}/{rec.filename}", headers, rec.length)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def ingest(self, pattern: str, limit: int = 10, role: str = "competitor") -> list[SourceDoc]:
        docs = []
        for rec in self.search(pattern, limit):
            try:
                parsed = parse_warc(self.fetch(rec))
            except (ValueError, OSError, httpx.HTTPError) as e:
                self.errors.append(f"{rec.url}: {type(e).__name__}: {e}")
                continue
            if parsed is None or parsed[2] != 200:
                continue
            uri, html, _ = parsed
            prov = {
                "source": "commoncrawl",
                "index": self.index_name,
                "warc": rec.filename,
                "offset": rec.offset,
                "length": rec.length,
                "timestamp": rec.timestamp,
            }
            docs.append(extract(html, uri or rec.url, role=role, provenance=prov))  # type: ignore[arg-type]
        return docs
