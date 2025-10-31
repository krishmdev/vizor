import io

import httpx
import pytest

from vizor.ingest.commoncrawl import CommonCrawl, parse_cdx, parse_warc

CDX = (
    '{"url": "https://crema-lab.example/aria", "filename": "crawl-data/x.warc.gz", '
    '"offset": "100", "length": "2000", "status": "200", "mime": "text/html", '
    '"timestamp": "20260301000000"}\n'
    "__import__('os').system('echo pwned')\n"
    '{"url": "https://b.example/", "filename": "f", "offset": "1", "length": "2"}\n'
)

HTML = (
    b"<html><head><title>Aria</title></head><body><main><p>The Aria is a single boiler "
    b"espresso machine with PID temperature control, pre-infusion and a 58 mm group. It heats "
    b"up in about five minutes and holds temperature within a degree over a session.</p>"
    b"</main></body></html>"
)


def _warc(uri: str, body: bytes, status: str = "200 OK") -> bytes:
    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter

    buf = io.BytesIO()
    w = WARCWriter(buf, gzip=True)
    headers = StatusAndHeaders(
        status, [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.1"
    )
    w.write_record(
        w.create_warc_record(uri, "response", payload=io.BytesIO(body), http_headers=headers)
    )
    return buf.getvalue()


def test_cdx_is_parsed_as_json_never_evaluated():
    recs = parse_cdx(CDX)
    assert [r.url for r in recs] == ["https://crema-lab.example/aria", "https://b.example/"]
    assert recs[0].offset == 100 and recs[0].length == 2000


def test_warc_round_trip():
    uri, html, status = parse_warc(_warc("https://crema-lab.example/aria", HTML))
    assert uri == "https://crema-lab.example/aria" and status == 200 and "single boiler" in html


def test_ingest_with_mocked_transport(tmp_path):
    warc = _warc("https://crema-lab.example/aria", HTML)
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/collinfo.json":
            return httpx.Response(200, json=[{"id": "CC-MAIN-2026-10"}, {"id": "CC-MAIN-2025-51"}])
        if req.url.path.endswith("-index"):
            seen["params"] = dict(req.url.params.multi_items())
            return httpx.Response(200, text=CDX.splitlines()[0] + "\n")
        seen["range"] = req.headers["range"]
        return httpx.Response(206, content=warc)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cc = CommonCrawl(index="latest", cache_dir=tmp_path, min_interval_s=0, client=client)
    assert cc.index_name == "CC-MAIN-2026-10"
    docs = cc.ingest("crema-lab.example/*", limit=5)
    assert seen["range"] == "bytes=100-2099"
    assert seen["params"]["output"] == "json"
    assert len(docs) == 1 and docs[0].provenance["source"] == "commoncrawl"
    assert "single boiler" in docs[0].body
    # second fetch comes from the WARC cache
    assert list((tmp_path / "warc").glob("*.warc.gz"))


@pytest.mark.network
def test_live_common_crawl_smoke(tmp_path):
    cc = CommonCrawl(cache_dir=tmp_path)
    assert cc.index_name.startswith("CC-MAIN-")
    docs = cc.ingest("commoncrawl.org/*", limit=2)
    assert all(d.url for d in docs)


def test_range_request_must_return_206(tmp_path):
    warc = _warc("https://crema-lab.example/aria", HTML)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("-index"):
            lines = [CDX.splitlines()[0], CDX.splitlines()[0].replace("x.warc.gz", "y.warc.gz")]
            return httpx.Response(200, text="\n".join(lines))
        if "y.warc.gz" in req.url.path:
            return httpx.Response(200, content=warc)  # ignored the Range header
        return httpx.Response(206, content=warc)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cc = CommonCrawl(index="CC-MAIN-2026-10", cache_dir=tmp_path, min_interval_s=0, client=client)
    docs = cc.ingest("crema-lab.example/*")
    assert len(docs) == 1
    assert len(cc.errors) == 1 and "206" in cc.errors[0]
