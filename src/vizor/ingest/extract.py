"""HTML -> SourceDoc.

Title/canonical/meta/headings come from the page head (ported from the old HTMLParser), body text
from trafilatura, JSON-LD from extruct (ported from StructuredDataExtractor), and FAQ pairs from
FAQPage JSON-LD or from on-page question headings followed by an answer paragraph.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from vizor.types import Role, SourceDoc, stable_id


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


def jsonld_items(html: str, base_url: str) -> list[dict]:
    try:
        import extruct

        data = extruct.extract(html, base_url=base_url, syntaxes=["json-ld"], uniform=True)
        items = data.get("json-ld", [])
    except Exception:
        items = []
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                items.append(json.loads(tag.string or ""))
            except ValueError:
                continue
    flat: list[dict] = []
    for it in items:
        if isinstance(it, dict) and "@graph" in it:
            flat.extend(x for x in it["@graph"] if isinstance(x, dict))
        elif isinstance(it, dict):
            flat.append(it)
    return flat


def schema_types(items: list[dict]) -> list[str]:
    types: set[str] = set()
    for it in items:
        t = it.get("@type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            types.update(str(x) for x in t)
    return sorted(types)


def faq_from_jsonld(items: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for it in items:
        if "FAQPage" not in schema_types([it]):
            continue
        for q in it.get("mainEntity", []) or []:
            if not isinstance(q, dict):
                continue
            ans = q.get("acceptedAnswer") or {}
            if isinstance(ans, list):
                ans = ans[0] if ans else {}
            name, text = q.get("name", ""), ans.get("text", "") if isinstance(ans, dict) else ""
            if name and text:
                pairs.append((name.strip(), BeautifulSoup(text, "lxml").get_text(" ").strip()))
    return pairs


def faq_from_headings(soup: BeautifulSoup) -> list[tuple[str, str]]:
    pairs = []
    for h in soup.find_all(["h2", "h3", "h4", "dt", "summary"]):
        q = _text(h)
        if not q.endswith("?") or len(q.split()) > 20:
            continue
        nxt = h.find_next_sibling(["p", "dd", "div"])
        if nxt is None and h.name == "summary":
            nxt = h.parent
        a = _text(nxt)
        if a and a != q:
            pairs.append((q, a.removeprefix(q).strip()))
    return pairs


def extract(
    html: str, url: str, role: Role = "competitor", provenance: dict | None = None
) -> SourceDoc:
    soup = BeautifulSoup(html, "lxml")
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    title = _text(soup.find("title")) or _text(soup.find("h1"))
    meta = soup.find("meta", attrs={"name": "description"})
    description = (meta.get("content") or "").strip() if meta else ""
    canonical = soup.find("link", rel="canonical")
    canonical_url = canonical.get("href") if canonical and canonical.get("href") else url
    headings = tuple(_text(h) for h in soup.find_all(["h1", "h2", "h3"]) if _text(h))

    items = jsonld_items(html, canonical_url)
    faq = faq_from_jsonld(items) or faq_from_headings(soup)

    body = None
    try:
        import trafilatura

        body = trafilatura.extract(
            html, favor_precision=True, include_tables=True, include_comments=False, url=url
        )
    except Exception:
        body = None
    if not body or len(body.split()) < 40:
        main = soup.find("main") or soup.find("article") or soup.body or soup
        for bad in main.find_all(["script", "style", "nav", "header", "footer"]):
            bad.decompose()
        body = "\n".join(_text(p) for p in main.find_all(["p", "li", "td"]) if _text(p))
    faq_text = {x for pair in faq for x in pair}
    paras = [ln.strip() for ln in body.split("\n") if ln.strip() and ln.strip() not in faq_text]
    paras = [p for p in paras if p not in headings]

    links = []
    scope = soup.find("main") or soup.find("article") or soup.body or soup
    for a in scope.find_all("a", href=True):
        href = urljoin(canonical_url, a["href"])
        if urlparse(href).netloc.lower().removeprefix("www.") == domain and href != canonical_url:
            anchor = _text(a)
            if anchor and (anchor, href) not in links:
                links.append((anchor, href))

    return SourceDoc(
        doc_id=stable_id(canonical_url),
        url=canonical_url,
        domain=domain,
        role=role,
        title=title,
        meta_description=description,
        headings=headings,
        body="\n\n".join(paras),
        faq=tuple(faq),
        jsonld=tuple(items),
        links=tuple(links),
        provenance=provenance or {},
    )
