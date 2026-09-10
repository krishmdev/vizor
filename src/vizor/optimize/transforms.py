"""Page-side optimization arms. Each returns a new document version and a short diff.

Deterministic arms only rearrange or restate facts already on the page (or, for internal links,
point at real pages in the corpus). The JSON-LD shapes are ported from the old action handlers,
but every field is filled from extracted page facts rather than templates, and nothing is
invented: a Product gets an Offer only if a price appears on the page.

The LLM rewrite arms use prompts adapted from GEO's geo_functions.py (Apache-2.0, see NOTICE).
`cite_sources` and `statistics_addition` are the paper's exact methods; their prompts tell the
model it may invent sources and numbers, so they are flagged `fabrication_risk` and only run with
--allow-fabrication.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from vizor.attribution.citations import split_sentences
from vizor.embed import Embedder
from vizor.generate.llm import LLM
from vizor.types import Query, SourceDoc

STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "best",
        "buy",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "to",
        "under",
        "vs",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "you",
        "your",
        "dollars",
        "use",
        "using",
        "often",
        "long",
        "much",
        "many",
        "get",
        "make",
        "fix",
        "take",
        "taking",
        "time",
        "good",
        "cheap",
    ]
)


@dataclass
class TransformContext:
    """What a transform may look at: the corpus, the tracked queries, and an embedder."""

    docs: dict[str, SourceDoc]
    queries: Sequence[Query]
    embedder: Embedder
    llm: LLM | None = None
    n_context_queries: int = 8
    # Shared log of grounding-guard decisions for LLM rewrites ({transform, doc_id, accepted,
    # violations}), so a run can report how often a rewrite was rejected.
    events: list = field(default_factory=list)
    # Sampling settings of the rewriter's calls (the rewriter config's temperature, seed and
    # max_tokens when the run has one).
    rewrite: dict = field(
        default_factory=lambda: {"temperature": 0.0, "seed": 0, "max_tokens": 1500}
    )
    # Frozen rewrites (doc_id -> the rewriter's raw output), read from a pre-generated
    # rewrites.json: the transform then makes no call and applies the same guard to that text.
    frozen: dict[str, str] | None = None
    _qvec: np.ndarray | None = field(default=None, repr=False)
    _df: Counter | None = field(default=None, repr=False)

    def query_vectors(self) -> np.ndarray:
        if self._qvec is None:
            self._qvec = self.embedder.encode([q.text for q in self.queries], kind="query")
        return self._qvec

    def doc_vector(self, doc: SourceDoc) -> np.ndarray:
        v = self.embedder.encode([doc.title + ". " + doc.body[:2000]], kind="passage")[0]
        return v / (np.linalg.norm(v) or 1.0)

    def ngram_df(self) -> Counter:
        """In how many corpus pages each 1-3-gram appears (for the page-vs-corpus TF-IDF)."""
        if self._df is None:
            self._df = Counter(
                g for d in self.docs.values() for g in set(_term_grams(_page_text(d)))
            )
        return self._df

    def context_queries(self, doc: SourceDoc) -> list[Query]:
        """The tracked queries this page is most relevant to."""
        sims = self.query_vectors() @ self.doc_vector(doc)
        top = np.argsort(-sims, kind="stable")[: self.n_context_queries]
        return [self.queries[i] for i in top]


@dataclass(frozen=True)
class Transform:
    name: str
    kind: str  # "doc"
    apply: Callable[[SourceDoc, TransformContext], tuple[SourceDoc, str]]
    requires_llm: bool = False
    fabrication_risk: bool = False
    description: str = ""
    uses_queries: bool = False  # reads the tracked queries, so it must be cross-fitted


def _sentences(doc: SourceDoc) -> list[str]:
    return [s for p in doc.paragraphs for s in split_sentences(p) if len(s.split()) >= 4]


def _central(doc: SourceDoc, ctx: TransformContext, k: int) -> list[str]:
    sents = _sentences(doc)
    if not sents:
        return []
    v = ctx.embedder.encode(sents)
    centroid = v.mean(axis=0)
    order = np.argsort(-(v @ centroid), kind="stable")[:k]
    return [sents[i] for i in sorted(order)]


def _ngrams(text: str, n_max: int = 3) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for n in range(1, n_max + 1):
        for i in range(len(toks) - n + 1):
            g = toks[i : i + n]
            if g[0] not in STOP and g[-1] not in STOP:
                out.append(" ".join(g))
    return out


def keyphrases(doc: SourceDoc, ctx: TransformContext, k: int = 5) -> list[str]:
    """c-TF-IDF over the page's context queries: phrases frequent in those queries, rare across
    all tracked queries, and actually present on the page."""
    page = " ".join(re.findall(r"[a-z0-9]+", (doc.title + " " + doc.body).lower()))
    local = Counter(g for q in ctx.context_queries(doc) for g in _ngrams(q.text))
    df = Counter(g for q in ctx.queries for g in set(_ngrams(q.text)))
    n = len(ctx.queries)
    scored = {
        g: c * math.log(1 + n / df[g]) * (1 + 0.5 * g.count(" "))
        for g, c in local.items()
        if f" {g} " in f" {page} "
    }
    out: list[str] = []
    for g, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0])):
        if not any(g in o or o in g for o in out):
            out.append(g)
        if len(out) == k:
            break
    return out


def _site_name(domain: str) -> str:
    return domain.split(".")[0].replace("-", " ").title()


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def noop(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    return doc, ""


def metadata(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """Title = H1 + the page's top query keyphrase; description = lead sentence plus the most
    central other sentence that fits in 155 characters."""
    h1 = doc.headings[0] if doc.headings else doc.title
    phrase = next(iter(keyphrases(doc, ctx, 1)), "")
    title = h1 if not phrase or phrase in h1.lower() else f"{h1}: {phrase.title()}"
    title = _clip(title, 60)
    sents = _sentences(doc)
    lead = sents[0] if sents else doc.title
    desc = _clip(lead, 155)
    for s in _central(doc, ctx, 3):
        if s != lead and len(f"{desc} {s}") <= 155:
            desc = f"{desc} {s}"
            break
    new = doc.evolve("metadata", title=title, meta_description=desc)
    return (
        new,
        f"title: {doc.title!r} -> {title!r}\ndescription: {doc.meta_description!r} -> {desc!r}",
    )


def _question(q: str) -> str:
    q = q.strip().rstrip("?")
    low = q.lower()
    if re.match(r"^(what|how|why|when|where|which|who|is|are|can|does|do|should)\b", low):
        out = q
    elif " vs " in low:
        a, b = re.split(r"\s+vs\s+", q, maxsplit=1)
        out = f"How does the {a} compare with the {b}"
    elif low.endswith(" price"):
        out = f"How much does the {q[:-6]} cost"
    elif low.startswith("best "):
        out = f"What is the {q}"
    elif low.startswith(("buy ", "where to buy ")):
        out = f"Where can I {q.removeprefix('where to ')}"
    else:
        out = f"What should I know about {q}"
    return out[0].upper() + out[1:] + "?"


def faq_rewrite(doc: SourceDoc, ctx: TransformContext, k: int = 4) -> tuple[SourceDoc, str]:
    """Q = a tracked query the page can actually answer; A = the page's best-matching sentence.
    A query qualifies only if its best sentence here scores above the median best-sentence score
    over all tracked queries, which keeps off-topic questions out without a fixed threshold."""
    sents = _sentences(doc)
    if not sents:
        return doc, ""
    sv = ctx.embedder.encode(sents)
    sims = ctx.query_vectors() @ sv.T  # (queries, sentences)
    bar = float(np.median(sims.max(axis=1)))
    index = {q.query_id: i for i, q in enumerate(ctx.queries)}
    pairs = []
    used: set[int] = set()
    for q in ctx.context_queries(doc):
        row = sims[index[q.query_id]]
        order = [int(i) for i in np.argsort(-row, kind="stable") if int(i) not in used]
        if not order or row[order[0]] <= bar:
            continue
        best = [order[0]]
        if len(order) > 1 and row[order[1]] >= 0.9 * row[order[0]]:
            best.append(order[1])
        used.update(best)
        pairs.append((_question(q.text), " ".join(sents[i] for i in sorted(best))))
        if len(pairs) == k:
            break
    if not pairs:
        return doc, ""
    new = doc.evolve("faq_rewrite", faq=(*doc.faq, *pairs))
    return new, "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)


_PRICE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")


def jsonld_insert(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    name = doc.headings[0] if doc.headings else doc.title
    brand = _site_name(doc.domain)
    summary = " ".join(_central(doc, ctx, 1))
    price = _PRICE.search(doc.body)
    items: list[dict] = []
    if price:
        items.append(
            {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": name if brand.lower() in name.lower() else f"{brand} {name}",
                "brand": {"@type": "Brand", "name": brand},
                "description": _clip(summary, 300),
                "url": doc.url,
                "offers": {
                    "@type": "Offer",
                    "price": price.group(1).replace(",", ""),
                    "priceCurrency": "USD",
                    "url": doc.url,
                },
            }
        )
    else:
        items.append(
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": _clip(name, 110),
                "description": _clip(summary, 300),
                "author": {"@type": "Organization", "name": brand},
                "url": doc.url,
            }
        )
    if doc.faq:
        items.append(
            {
                "@context": "https://schema.org",
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": q,
                        "acceptedAnswer": {"@type": "Answer", "text": a},
                    }
                    for q, a in doc.faq
                ],
            }
        )
    new = doc.evolve("jsonld_insert", jsonld=(*doc.jsonld, *items))
    return new, "added " + ", ".join(i["@type"] for i in items)


def internal_links(doc: SourceDoc, ctx: TransformContext, k: int = 3) -> tuple[SourceDoc, str]:
    same = [d for d in ctx.docs.values() if d.domain == doc.domain and d.doc_id != doc.doc_id]
    if not same:
        return doc, ""
    v = ctx.doc_vector(doc)
    ranked = sorted(same, key=lambda d: -float(ctx.doc_vector(d) @ v))[:k]
    links = [(d.headings[0] if d.headings else d.title, d.url) for d in ranked]
    links = [x for x in links if x not in doc.links]
    new = doc.evolve("internal_links", links=(*doc.links, *links))
    return new, "\n".join(f"{a} -> {h}" for a, h in links)


_NUMERIC = re.compile(r"\d")
_QUOTE = re.compile(r"[“\"][^”\"]{12,}[”\"]")


def _surface(
    doc: SourceDoc, pick: Callable[[str], bool], name: str, limit: int = 6
) -> tuple[SourceDoc, str]:
    moved: list[str] = []
    kept_paras: list[str] = []
    for p in doc.paragraphs:
        stay = []
        for s in split_sentences(p):
            if pick(s) and len(moved) < limit:
                moved.append(s)
            else:
                stay.append(s)
        if stay:
            kept_paras.append(" ".join(stay))
    if not moved:
        return doc, ""
    lead, rest = (kept_paras[:1], kept_paras[1:]) if kept_paras else ([], [])
    body = "\n\n".join([*lead, " ".join(moved), *rest])
    return doc.evolve(name, body=body), "moved up:\n" + "\n".join(moved)


def stats_surface(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """GEO's Statistics Addition, grounded: move the page's own numbers up instead of adding any."""
    return _surface(doc, lambda s: bool(_NUMERIC.search(s)), "stats_surface")


def quote_surface(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """GEO's Quotation Addition, grounded: move the page's existing quotes up."""
    return _surface(doc, lambda s: bool(_QUOTE.search(s)), "quote_surface")


def keyword_stuffing(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """The paper's traditional-SEO control: append the page's top query keywords."""
    kws = keyphrases(doc, ctx, 10)
    line = "Related searches: " + ", ".join(kws) + "."
    return doc.evolve("keyword_stuffing", body=doc.body + "\n\n" + line), line


# ------------------------------------------------------------------ Study 2 arms (query-blind)
# Units a key fact or a guarded quantity can carry. Longer forms come first (km/h before km, wh
# before w); bare letters that are common words ("in", "m", "l", "x") are left out.
_UNIT = (
    r"(?:km/h|kg|km|lbs?|psi|bar|mm|cm|mph|wh|w|nm|lumens?|%|percent|litres?|liters?|"
    r"inch(?:es)?|hours?|minutes?|seconds?|°\s?c|db|cycles|g)"
)
_KEY_FACT = re.compile(
    rf"\$\s?\d|\b\d[\d,.]*\s?{_UNIT}(?![a-z])|\b\d+\s?-\s?speed\b|\b\d+\s?x\s?\d+", re.I
)
MAX_LEAD_FACTS = 4
MAX_LEAD_WORDS = 90


def key_fact(sentence: str) -> bool:
    """A sentence states a key fact if it has a price, a number with a unit, or a spec pattern
    such as "2 x 10" or "7-speed"."""
    return bool(_KEY_FACT.search(sentence))


def answer_first(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """Move the page's key-fact sentences (prices, numbers with units, specs; at most 4 and 90
    words, in page order) into a new first paragraph, so they land in the lead body window.
    Nothing is added or reworded, and it reads no queries."""
    picked: list[str] = []
    words = 0
    for p in doc.paragraphs:
        for sent in split_sentences(p):
            n = len(sent.split())
            if key_fact(sent) and len(picked) < MAX_LEAD_FACTS and words + n <= MAX_LEAD_WORDS:
                picked.append(sent)
                words += n
    if not picked:
        return doc, ""
    kept = []
    for p in doc.paragraphs:
        stay = [x for x in split_sentences(p) if x not in picked]
        if stay:
            kept.append(" ".join(stay))
    body = "\n\n".join([" ".join(picked), *kept])
    return doc.evolve("answer_first", body=body), "moved to the lead:\n" + "\n".join(picked)


_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_CAP = re.compile(r"\b[A-Z][A-Za-z0-9&'’-]*")
_QTY = re.compile(rf"(\d+(?:[.,]\d+)*)\s?-?\s?({_UNIT})(?![a-z])", re.I)
_PRICE_QTY = re.compile(r"\$\s?(\d+(?:[.,]\d+)*)")
_UNIT_ALIAS = {
    "lb": "lbs",
    "lumen": "lumens",
    "percent": "%",
    "litre": "l",
    "litres": "l",
    "liter": "l",
    "liters": "l",
    "inches": "inch",
    "hour": "hours",
    "minute": "minutes",
    "second": "seconds",
}
NUMBER_WORDS = set(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "dozen",
        "twice",
        "double",
        "triple",
        "half",
        "quarter",
    ]
)
# Words that often start a sentence in a rewrite without being names.
STARTERS = STOP | set(
    [
        "this",
        "these",
        "that",
        "those",
        "it",
        "its",
        "our",
        "we",
        "they",
        "there",
        "here",
        "then",
        "also",
        "additionally",
        "however",
        "moreover",
        "furthermore",
        "overall",
        "finally",
        "first",
        "second",
        "third",
        "next",
        "plus",
        "note",
        "unlike",
        "compared",
        "like",
        "each",
        "every",
        "both",
        "most",
        "many",
        "some",
        "all",
        "because",
        "while",
        "if",
        "but",
        "so",
        "after",
        "before",
    ]
)
MIN_WORDS_KEPT = 0.7


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def _quantities(text: str) -> set[tuple[str, str]]:
    """(number, unit) pairs, e.g. ("160", "psi") and ("49", "$")."""
    out = set()
    for n, u in _QTY.findall(text):
        u = re.sub(r"\s", "", u.lower())
        out.add((n.replace(",", "").rstrip("."), _UNIT_ALIAS.get(u, u)))
    out |= {(n.replace(",", "").rstrip("."), "$") for n in _PRICE_QTY.findall(text)}
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in STOP}


def grounding_violations(new: str, old: str) -> list[str]:
    """What in `new` is not supported by `old`:
    - a number that is not in `old` (thousands separators ignored);
    - a number with a unit (or a price) whose pairing is not in `old`, so "160 psi" cannot
      become "160 bar";
    - a spelled-out number word ("two", "dozen", "twice") that `old` does not use;
    - a name: a capitalized word after the start of a sentence that `old` does not have with
      the same capitalization, or a sentence-initial capitalized word that `old` lacks in any
      case, is not a common sentence starter, and looks like a name (it is followed by another
      capitalized word, or appears capitalized mid-sentence in `new`)."""
    bad = sorted(_numbers(new) - _numbers(old))
    bad += [f"{n} {u}" for n, u in sorted(_quantities(new) - _quantities(old))]
    old_lower_words = set(re.findall(r"[a-z]+", old.lower()))
    bad += sorted(
        w
        for w in set(re.findall(r"[a-z]+", new.lower())) & NUMBER_WORDS
        if w not in old_lower_words
    )
    old_words = set(re.findall(r"[A-Za-z0-9&'’-]+", old))
    old_lower = {w.lower() for w in old_words}
    sentences = split_sentences(new.replace("\n", " "))
    mid_caps = set()
    for sent in sentences:
        lead = len(sent) - len(sent.lstrip("\"'“([*-• "))
        mid_caps |= {m.group(0) for m in _CAP.finditer(sent) if m.start() != lead}
    for sent in sentences:
        lead = len(sent) - len(sent.lstrip("\"'“([*-• "))
        caps = list(_CAP.finditer(sent))
        for i, m in enumerate(caps):
            w = m.group(0).rstrip("'’-")
            if w in old_words:
                continue
            if m.start() == lead:
                if w.lower() in old_lower or w.lower() in STARTERS:
                    continue
                nxt = caps[i + 1] if i + 1 < len(caps) else None
                joined = nxt is not None and sent[m.end() : nxt.start()].strip() == ""
                if not (joined or w in mid_caps):
                    continue
            bad.append(w)
    return list(dict.fromkeys(bad))


MIN_NUMBERS_KEPT = 0.9
EVIDENCE_PROMPT = (
    "Rewrite the following web page so that the numbers it already contains (prices, weights, "
    "sizes, capacities, run times, speeds, percentages) are stated early and explicitly, in "
    "short sentences near the start of the paragraphs where they belong. Use only the numbers, "
    "names and facts that appear in the page. Do not add any new number, name, product, "
    "statistic, source or claim, and do not remove any fact. Keep the paragraph structure."
)


def rewriter_fingerprint(rc) -> dict:
    """What decides a rewrite besides the page: the rewriter's model and request settings and
    the prompt. A run only reuses frozen rewrites whose fingerprint equals its own."""
    import hashlib

    return {
        "model": rc.model,
        "preset": rc.server_meta.get("preset"),
        "temperature": rc.temperature,
        "seed": rc.seed,
        "max_tokens": rc.max_tokens,
        "request": rc.request_extra(),
        "prompt_sha256": hashlib.sha256((GEO_SYSTEM + EVIDENCE_PROMPT).encode()).hexdigest(),
    }


def evidence_messages(doc: SourceDoc) -> list[dict[str, str]]:
    user = (
        f"{EVIDENCE_PROMPT}\n\nReturn only the rewritten page text, with paragraphs separated "
        f"by blank lines and no commentary.\n\nPage:\n```\n{doc.body}\n```"
    )
    return [{"role": "system", "content": GEO_SYSTEM}, {"role": "user", "content": user}]


def rewrite_change(before: str, after: str) -> str:
    """How a rewrite changed a page's text: "unchanged" (same sentences in the same order, up
    to whitespace), "reordered" (the same sentences in another order) or "changed". The guard
    accepts all three; this flags the first two so accepted no-ops are not read as edits."""

    def sents(t: str) -> list[str]:
        return [" ".join(x.split()) for x in split_sentences(" ".join(t.split())) if x.strip()]

    a, b = sents(before), sents(after)
    if a == b:
        return "unchanged"
    if sorted(a) == sorted(b):
        return "reordered"
    return "changed"


def evidence_surface_llm(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """GEO's statistics addition restricted to numbers already on the page: an LLM restates the
    page with its own numbers surfaced. A grounding guard rejects any output with a number or
    name that is not on the page (see `grounding_violations`), or that keeps fewer than 90% of
    the page's distinct numbers or 70% of its distinct content words; a rejected page keeps its
    original text."""
    info: dict = {}
    if ctx.frozen is not None:
        if doc.doc_id not in ctx.frozen:
            raise RuntimeError(f"no frozen rewrite for {doc.doc_id}")
        text = ctx.frozen[doc.doc_id]
        info["frozen"] = True
    else:
        if ctx.llm is None:
            raise RuntimeError("evidence_surface_llm needs a rewriter LLM")
        c = ctx.llm.complete(
            evidence_messages(doc),
            temperature=ctx.rewrite["temperature"],
            seed=ctx.rewrite["seed"],
            max_tokens=ctx.rewrite["max_tokens"],
        )
        text = c.text
        info = {
            "finish_reason": c.usage.get("finish_reason"),
            "thinking_tokens": c.usage.get("thinking_tokens"),
            "completion_tokens": c.usage.get("completion_tokens"),
            "reasoning": c.reasoning,
        }
    info["text"] = text
    # Only the final page text is used: a thinking model's block never reaches the guard.
    text = re.sub(r"^\s*<think>.*?</think>", "", text, flags=re.S)
    body = re.sub(r"^```[\w-]*\s*\n|\n?```\s*$", "", text.strip()).strip()
    # Everything the page itself says: body, title, description, headings and FAQ.
    source = "\n".join(
        [
            doc.body,
            doc.title,
            doc.meta_description,
            *doc.headings,
            *(f"{q} {a}" for q, a in doc.faq),
        ]
    )
    bad = grounding_violations(body, source) if body else ["<empty>"]
    # Surfacing must not become cutting: most of the page's own numbers and words have to survive.
    had, kept = _numbers(doc.body), _numbers(body) & _numbers(doc.body)
    if had and len(kept) < MIN_NUMBERS_KEPT * len(had):
        bad.append(f"<kept {len(kept)} of {len(had)} numbers>")
    words, kept_w = _content_words(doc.body), _content_words(body) & _content_words(doc.body)
    if words and len(kept_w) < MIN_WORDS_KEPT * len(words):
        bad.append(f"<kept {len(kept_w)} of {len(words)} words>")
    ctx.events.append(
        {
            "transform": "evidence_surface_llm",
            "doc_id": doc.doc_id,
            "accepted": not bad,
            "violations": bad[:10],
            "change": rewrite_change(doc.body, body) if body else "changed",
            **info,
        }
    )
    if bad:
        return doc, ""
    return doc.evolve("evidence_surface_llm", body=body), (
        f"rewrote {len(doc.body.split())} -> {len(body.split())} words"
    )


_WH = re.compile(r"^(what|how|why|when|where|which|who|is|are|can|does|do|should)\b", re.I)
MAX_FAQ_V2 = 2


def _heading_question(heading: str, product: str | None) -> str:
    """ "Battery life" on a product page -> "What should I know about battery life on the
    Larkspur Beam 800?"; on a guide -> "What should I know about battery life?"."""
    h = heading.strip().rstrip("?.:")
    if _WH.match(h):
        return h[0].upper() + h[1:] + "?"
    tail = f" on the {product}" if product else ""
    return f"What should I know about {h[0].lower() + h[1:]}{tail}?"


def faq_rewrite_v2(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """A query-blind FAQ: questions come from the page's own section headings (not from tracked
    queries), each answer is the single page sentence that best matches its question, no
    sentence answers two questions, and there are at most 2 pairs (the two best-matched)."""
    sents = _sentences(doc)
    heads = [h for h in doc.headings[1:] if h.strip()]
    if not sents or not heads:
        return doc, ""
    h1 = doc.headings[0] if doc.headings else doc.title
    product = h1 if _site_name(doc.domain).lower() in h1.lower() else None
    qs = [_heading_question(h, product) for h in heads]
    sims = ctx.embedder.encode(qs, kind="query") @ ctx.embedder.encode(sents).T
    order = sorted(range(len(qs)), key=lambda i: (-float(sims[i].max()), i))
    pairs: list[tuple[str, str]] = []
    used: set[int] = set()
    for i in order:
        best = [int(j) for j in np.argsort(-sims[i], kind="stable") if int(j) not in used]
        if not best:
            continue
        used.add(best[0])
        pairs.append((qs[i], sents[best[0]]))
        if len(pairs) == MAX_FAQ_V2:
            break
    if not pairs:
        return doc, ""
    new = doc.evolve("faq_rewrite_v2", faq=(*doc.faq, *pairs))
    return new, "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)


# Study 3 arms: query-blind edits aimed at passage selection and retrieval rank. All three are
# deterministic, read only the page (and, for retrieval_meta, corpus term frequencies), and run
# their output through the same grounding guard as the LLM rewrites, logging the verdict and
# whether the page actually changed.
FACT_PASSAGE_WORDS = 110
MAX_ANCHOR_ADDED = 16
# Words that can stand for the page's own product at the start of a sentence ("The pump", "This
# bike"). One only counts if the page's first sentence uses it, so "The bike" on a lock page is
# left alone.
PRODUCT_NOUNS = (
    "bike",
    "e-bike",
    "light",
    "pump",
    "lock",
    "helmet",
    "bag",
    "pannier",
    "trainer",
    "model",
    "product",
)


# Verbs after which "The <one to three lowercase words>" at the start of a product page's
# sentence is read as a part of the product ("The dual head fits", "The shackle is").
PART_VERBS = (
    "is|are|has|have|takes|fits|holds|weighs|runs|uses|gives|comes|measures|handles|charges"
    "|reaches|stops|lasts"
)


def _page_text(doc: SourceDoc) -> str:
    return "\n".join([doc.title, *doc.headings, doc.body, *(f"{q} {a}" for q, a in doc.faq)])


def _guard_source(doc: SourceDoc) -> str:
    return "\n".join(
        [
            doc.body,
            doc.title,
            doc.meta_description,
            *doc.headings,
            *(f"{q} {a}" for q, a in doc.faq),
        ]
    )


@dataclass(frozen=True)
class _Entity:
    brand: str  # "Larkspur"
    name: str  # "Larkspur Beam 800", or "Larkspur Tire Pressure guide" on a guide page
    short: str | None  # "Beam 800" on a product page, None on a guide
    nouns: tuple[str, ...]  # PRODUCT_NOUNS the page's first sentence uses
    products: tuple[str, ...]  # short names of every product page on the same site


def _entity(doc: SourceDoc, ctx: TransformContext) -> _Entity:
    brand = _site_name(doc.domain)
    h1 = doc.headings[0] if doc.headings else doc.title

    def short_of(h: str) -> str | None:
        if h.lower().startswith(brand.lower() + " ") and len(h) > len(brand) + 1:
            return h[len(brand) + 1 :].strip()
        return None

    products = sorted(
        {
            s
            for d in ctx.docs.values()
            if d.domain == doc.domain and (s := short_of(d.headings[0] if d.headings else d.title))
        },
        key=lambda x: (-len(x), x),
    )
    short = short_of(h1)
    first = (_sentences(doc) or [""])[0].lower()
    nouns = tuple(n for n in PRODUCT_NOUNS if re.search(rf"\b{re.escape(n)}\b", first))
    if short is None:
        return _Entity(brand, f"{brand} {h1} guide", None, (), tuple(products))
    return _Entity(brand, h1, short, (*nouns, "model", "product"), tuple(products))


def _brand_products(sent: str, ent: _Entity) -> str:
    """ "the Metro 7" -> "the Larkspur Metro 7" for every product of the site."""
    for p in ent.products:
        sent = re.sub(rf"(?<!{re.escape(ent.brand)} )\b{re.escape(p)}\b", f"{ent.brand} {p}", sent)
    return sent


def anchor_sentence(sent: str, ent: _Entity) -> str:
    """Replace a leading pronoun or bare reference to the page's product ("It", "Its", "The
    pump", "The Beam 800") with the product and brand name, name the product before a bare
    part reference ("The motor is" -> "On the Larkspur Haul, the motor is"), and brand any
    unbranded mention of the site's products. Guides only get the last step: their "it" is
    rarely the page."""
    if ent.short is not None:
        nouns = "|".join(re.escape(n) for n in (*ent.nouns, ent.short))
        for pat, rep in (
            (r"^It's\b", f"The {ent.name} is"),
            (r"^Its\b", f"The {ent.name}'s"),
            (r"^It\b", f"The {ent.name}"),
            (rf"^(?:The|This) (?:{nouns})\b(?!-)", f"The {ent.name}"),
            # a bare part reference: "The motor is ..." -> "On the Larkspur Haul, the motor is ..."
            (rf"^The (?=(?:[a-z][a-z-]* ){{0,2}}(?:{PART_VERBS})\b)", f"On the {ent.name}, the "),
        ):
            new = re.sub(pat, rep, sent, count=1)
            if new != sent:
                sent = new
                break
    return _brand_products(sent, ent)


def _names(sent: str, ent: _Entity) -> bool:
    if ent.short is None:
        return ent.brand.lower() in sent.lower()
    return ent.name.lower() in sent.lower()


def _lower_first(sent: str, doc: SourceDoc) -> str:
    """Lowercase the first word unless it looks like a name: capitalized somewhere on the page
    other than at the start of a sentence, or written with capitals or digits inside it."""
    m = re.match(r"[A-Z][a-z'’-]*\b", sent)
    if not m:
        return sent
    w = m.group(0)
    for s in _sentences(doc) + list(doc.headings[:1]):
        lead = len(s) - len(s.lstrip("\"'“([*-• "))
        if any(x.start() != lead for x in re.finditer(rf"\b{re.escape(w)}\b", s)):
            return sent
    return w.lower() + sent[len(w) :]


def _named_fact(sent: str, ent: _Entity, doc: SourceDoc) -> str:
    """A fact sentence that names the product and brand: anchored if it starts with a bare
    reference, otherwise prefixed ("For the Larkspur Beam 800, ...")."""
    out = anchor_sentence(sent, ent)
    if _names(out, ent):
        return out
    lead = f"For the {ent.name}, " if ent.short is not None else f"In the {ent.name}, "
    return lead + _lower_first(out, doc)


def _guarded(
    name: str, doc: SourceDoc, new: SourceDoc, ctx: TransformContext, before: str, after: str
) -> SourceDoc:
    """Run the grounding guard on the edited text and log the verdict and the kind of change;
    a page the guard rejects keeps its original version."""
    bad = grounding_violations(after, _guard_source(doc)) if after.strip() else []
    ctx.events.append(
        {
            "transform": name,
            "doc_id": doc.doc_id,
            "accepted": not bad,
            "violations": bad[:10],
            "change": rewrite_change(before, after) if before != after else "unchanged",
        }
    )
    return doc if bad else new


def fact_passage(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """One self-contained key-facts paragraph after the intro: the page's own sentences with a
    price, a number with a unit or a spec pattern (`key_fact`), in page order, each made to name
    the product and brand, up to 110 words so it fits one 120-word body window. The original
    sentences stay where they are. Reads no queries."""
    paras = doc.paragraphs
    if not paras:
        return doc, ""
    ent = _entity(doc, ctx)
    picked: list[str] = []
    words = 0
    for sent in (x for p in paras for x in split_sentences(p)):
        if not key_fact(sent):
            continue
        named = _named_fact(sent, ent, doc)
        n = len(named.split())
        if words + n <= FACT_PASSAGE_WORDS and named not in picked:
            picked.append(named)
            words += n
    if not picked:
        ctx.events.append(
            {
                "transform": "fact_passage",
                "doc_id": doc.doc_id,
                "accepted": True,
                "violations": [],
                "change": "unchanged",
            }
        )
        return doc, ""
    passage = " ".join(picked)
    body = "\n\n".join([paras[0], passage, *paras[1:]])
    new = _guarded("fact_passage", doc, doc.evolve("fact_passage", body=body), ctx, doc.body, body)
    if new is doc:
        return doc, ""
    return new, f"key-facts passage ({words} words) after the intro:\n{passage}"


def entity_anchor(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """Make each body window self-attributing: in the first sentence of every 120-word window,
    a leading pronoun or bare reference to the product becomes the product and brand name, and
    unbranded product names get the brand. Nothing else changes; at most 16 words are added per
    page (windows are edited in page order until the cap)."""
    from vizor.retrieve.chunk import window_starts

    paras = list(doc.paragraphs)
    ent = _entity(doc, ctx)
    added = 0
    notes: list[str] = []
    for pi, off in window_starts(paras):
        para = paras[pi]
        sents = split_sentences(para)
        # the first sentence that starts at or after the window's first word
        pos, target = 0, None
        for sent in sents:
            if pos >= off:
                target = sent
                break
            pos += len(sent.split())
        if target is None or target not in para:
            continue
        new = anchor_sentence(target, ent)
        extra = len(new.split()) - len(target.split())
        if new == target or added + extra > MAX_ANCHOR_ADDED:
            continue
        paras[pi] = para.replace(target, new, 1)
        added += extra
        notes.append(f"{target!r} -> {new!r}")
    body = "\n\n".join(paras)
    new_doc = doc.evolve("entity_anchor", body=body) if notes else doc
    new_doc = _guarded("entity_anchor", doc, new_doc, ctx, doc.body, body)
    if new_doc is doc:
        return doc, ""
    return new_doc, f"{len(notes)} window lead(s) anchored, +{added} words:\n" + "\n".join(notes)


# Words a key term may not contain, on top of STOP and STARTERS.
TERM_STOP = STARTERS | set(
    [
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "being",
        "will",
        "would",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "more",
        "less",
        "than",
        "about",
        "up",
        "down",
        "out",
        "over",
        "into",
        "onto",
        "off",
        "any",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "very",
        "just",
        "only",
        "even",
        "well",
        "same",
        "other",
        "own",
        "such",
        "still",
        "few",
        "lot",
        "way",
        "gets",
        "got",
        "makes",
        "gives",
        "uses",
        "keeps",
        "lets",
        "need",
        "needs",
        "want",
        "go",
        "goes",
        "come",
        "comes",
        "if",
        "us",
        "them",
        "their",
        "our",
        "we",
        "they",
        "usually",
        "often",
        "always",
        "almost",
        "really",
        "quickly",
        "easily",
        "better",
        "faster",
        "harder",
        "easier",
        "means",
        "feels",
        "feel",
        "keep",
        "put",
        "let",
        "give",
        "make",
        "who",
        "new",
        "add",
        "wrong",
        "covers",
        "stands",
        "included",
    ]
)


def _term_grams(text: str, n_max: int = 3) -> list[str]:
    """Alphabetic 1-3-grams inside punctuation-free spans (lines and sentences never join), with
    no function word in them."""
    out = []
    for span in re.split(r"[^A-Za-z \t'’-]+|\s['’-]|['’-]\s", text):
        toks = re.findall(r"[a-z][a-z'’-]*[a-z]|[a-z]", span.lower())
        for n in range(1, n_max + 1):
            for i in range(len(toks) - n + 1):
                g = toks[i : i + n]
                if not any(t in TERM_STOP for t in g) and len(g[-1]) > 2:
                    out.append(" ".join(g))
    return out


def page_terms(doc: SourceDoc, ctx: TransformContext, k: int = 8) -> list[str]:
    """The page's own key terms: alphabetic 1-3-grams (a multi-word one must occur at least
    twice) scored by TF-IDF of the page against the
    corpus (term count on the page x log(1 + pages / pages with the term), longer grams
    up-weighted), excluding the brand and the words of the H1, no two overlapping, in the page's
    own spelling."""
    text = _page_text(doc)
    tf = Counter(_term_grams(text))
    df = ctx.ngram_df()
    n = len(ctx.docs)
    brand = _site_name(doc.domain).lower()
    h1 = doc.headings[0] if doc.headings else doc.title
    skip = set(re.findall(r"[a-z0-9]+", (h1 + " " + brand).lower()))
    scored = {
        g: c * math.log(1 + n / max(df[g], 1)) * (1 + 0.5 * g.count(" "))
        for g, c in tf.items()
        if not set(g.split()) <= skip and brand not in g.split() and (c >= 2 or " " not in g)
    }
    out: list[str] = []
    for g, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0])):
        if not any(g in o or o in g for o in out):
            out.append(g)
        if len(out) == k:
            break
    surface = []
    for g in out:
        pat = r"\b" + r"\s+".join(map(re.escape, g.split())) + r"\b"
        m = re.search(pat, " ".join([*doc.headings, doc.body]), re.I)
        surface.append(m.group(0) if m else g)
    return surface


def _join_terms(terms: Sequence[str]) -> str:
    return terms[0] if len(terms) == 1 else ", ".join(terms[:-1]) + " and " + terms[-1]


def retrieval_meta(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """Title, meta description, H1 and H2s rebuilt from the page's own key terms (`page_terms`,
    TF-IDF of the page against the corpus; no query text):
    - title: H1 + the top three terms, at most 70 characters (terms dropped from the end);
    - H1: H1 + the top term;
    - each H2: the H2 + the unused term closest to it (bi-encoder cosine), one term each;
    - description: the page sentences that carry the most key terms, at most 155 characters.
    The body is not touched."""
    terms = page_terms(doc, ctx)
    if not terms:
        return doc, ""
    h1 = doc.headings[0] if doc.headings else doc.title
    title = f"{h1}: {_join_terms(terms[:3])}"
    for m in (2, 1):
        if len(title) <= 70:
            break
        title = f"{h1}: {_join_terms(terms[:m])}"
    title = _clip(title, 70)
    new_h1 = f"{h1}: {terms[0]}"
    heads = list(doc.headings[1:])
    pool = [t for t in terms[3:] if t.lower() not in title.lower()] or list(terms[1:])
    new_heads: list[str] = []
    if heads and pool:
        sims = ctx.embedder.encode(heads, kind="query") @ ctx.embedder.encode(pool).T
        used: set[int] = set()
        for i, h in enumerate(heads):
            free = [j for j in np.argsort(-sims[i], kind="stable") if int(j) not in used]
            if not free or pool[int(free[0])].lower() in h.lower():
                new_heads.append(h)
                continue
            used.add(int(free[0]))
            new_heads.append(f"{h}: {pool[int(free[0])]}")
    else:
        new_heads = heads
    low = [t.lower() for t in terms]
    sents = _sentences(doc)
    score = [sum(t in s.lower() for t in low) for s in sents]
    order = sorted(range(len(sents)), key=lambda i: (-score[i], i))
    chosen: list[int] = []
    for i in order:
        if score[i] == 0:
            break
        cand = " ".join(sents[j] for j in sorted([*chosen, i]))
        if len(cand) <= 155:
            chosen.append(i)
    desc = (
        " ".join(sents[j] for j in sorted(chosen))
        if chosen
        else _clip(sents[0], 155)
        if sents
        else ""
    )
    new = doc.evolve(
        "retrieval_meta",
        title=title,
        meta_description=desc or doc.meta_description,
        headings=(new_h1, *new_heads),
    )
    before = "\n".join([doc.title, doc.meta_description, *doc.headings])
    after = "\n".join([new.title, new.meta_description, *new.headings])
    new = _guarded("retrieval_meta", doc, new, ctx, before, after)
    if new is doc:
        return doc, ""
    return new, (
        f"terms: {', '.join(terms)}\ntitle: {doc.title!r} -> {new.title!r}\n"
        f"description: {doc.meta_description!r} -> {new.meta_description!r}\n"
        f"headings: {list(doc.headings)} -> {list(new.headings)}"
    )


# LLM rewrites. Prompts adapted from GEO (src/geo_functions.py).
GEO_SYSTEM = (
    "You are an expert in search engines and SEO. Websites are cited by language models that "
    "answer questions from several web sources. As the owner of one source, your task is to "
    "rewrite it so it is more visible in those answers, without changing its facts."
)
GEO_PROMPTS = {
    "fluency": "Rewrite the following source to make it more fluent without altering the core "
    "content. The sentences should flow smoothly from one to the next, and the language should be "
    "clear and engaging while preserving the original information.",
    "easy_to_understand": "Simplify the following source, using simple, easy-to-understand "
    "language while ensuring the key information is still conveyed. Do not omit, add, or alter "
    "any core information. The length of the new source should be about the same as the original.",
    "authoritative": "Transform the following source into an authoritative style without adding "
    "or removing any core information. The revised source should reflect confidence, expertise, "
    "and assertiveness, while keeping the original meaning. Keep the paragraph structure the same. "
    "No addition or deletion of facts is allowed.",
    "technical_terms": "Make the following source more technical, presenting the existing "
    "information with more precise technical terms, while ensuring the key information is still "
    "conveyed. Do not omit, add, or alter any core facts. Keep about the same length.",
    "cite_sources": "Revise the following source to include citations from credible sources. You "
    "may invent these sources but ensure they sound plausible. Citations should be in natural "
    "language, for example 'According to a recent industry report...'. Keep the core content and "
    "structure unaltered.",
    "statistics_addition": "Add positive, compelling statistics (even if hypothetical) at multiple "
    "relevant places in the text. Add them inline within existing sentences, and do not change "
    "other content.",
}
FABRICATING = {"cite_sources", "statistics_addition"}


def _rewrite(name: str) -> Callable[[SourceDoc, TransformContext], tuple[SourceDoc, str]]:
    def apply(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
        if ctx.llm is None:
            raise RuntimeError(f"{name} needs an LLM backend")
        user = (
            f"{GEO_PROMPTS[name]}\n\nReturn only the rewritten source text, with paragraphs "
            f"separated by blank lines and no commentary.\n\nSource:\n```\n{doc.body}\n```"
        )
        c = ctx.llm.complete(
            [{"role": "system", "content": GEO_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            seed=0,
            max_tokens=1500,
        )
        body = re.sub(r"^```[\w-]*\s*\n|\n?```\s*$", "", c.text.strip()).strip()
        return doc.evolve(
            name, body=body
        ), f"rewrote {len(doc.body.split())} -> {len(body.split())} words"

    return apply


TRANSFORMS: dict[str, Transform] = {
    t.name: t
    for t in [
        Transform("noop", "doc", noop, description="identity (A/A check)"),
        Transform(
            "metadata",
            "doc",
            metadata,
            description="title and meta description from page facts",
            uses_queries=True,
        ),
        Transform(
            "faq_rewrite",
            "doc",
            faq_rewrite,
            description="FAQ block answering tracked queries with page sentences",
            uses_queries=True,
        ),
        Transform(
            "jsonld_insert",
            "doc",
            jsonld_insert,
            description="Product/Article + FAQPage JSON-LD from page facts",
        ),
        Transform(
            "internal_links",
            "doc",
            internal_links,
            description="links to the 3 most similar pages on the same site",
        ),
        Transform(
            "stats_surface", "doc", stats_surface, description="move numeric sentences to the top"
        ),
        Transform(
            "quote_surface", "doc", quote_surface, description="move quoted sentences to the top"
        ),
        Transform(
            "keyword_stuffing",
            "doc",
            keyword_stuffing,
            description="append top query keywords (control)",
            uses_queries=True,
        ),
        Transform(
            "answer_first",
            "doc",
            answer_first,
            description="move key-fact sentences (prices, numbers with units) to the lead",
        ),
        Transform(
            "evidence_surface_llm",
            "doc",
            evidence_surface_llm,
            requires_llm=True,
            description="LLM restates the page's own numbers early; grounding guard",
        ),
        Transform(
            "faq_rewrite_v2",
            "doc",
            faq_rewrite_v2,
            description="up to 2 FAQ pairs from the page's headings, best-matching sentences",
        ),
        Transform(
            "fact_passage",
            "doc",
            fact_passage,
            description="one ~110-word key-facts passage after the intro, each sentence named",
        ),
        Transform(
            "entity_anchor",
            "doc",
            entity_anchor,
            description="window-lead pronouns and bare references -> product and brand name",
        ),
        Transform(
            "retrieval_meta",
            "doc",
            retrieval_meta,
            description="title, description, H1/H2 from the page's TF-IDF key terms",
        ),
        *[
            Transform(
                n,
                "doc",
                _rewrite(n),
                requires_llm=True,
                fabrication_risk=n in FABRICATING,
                description=f"GEO LLM rewrite: {n}",
            )
            for n in GEO_PROMPTS
        ],
    ]
}
DETERMINISTIC = [n for n, t in TRANSFORMS.items() if not t.requires_llm]
# The GEO paper's rewrites (`llm_rewrites: true`); evidence_surface_llm is only run by name.
LLM_REWRITES = [n for n in GEO_PROMPTS if n not in FABRICATING]


def apply_chain(
    doc: SourceDoc, arms: Sequence[str], ctx: TransformContext
) -> tuple[SourceDoc, str]:
    cur, notes = doc, []
    for arm in arms:
        cur, diff = TRANSFORMS[arm].apply(cur, ctx)
        if diff:
            notes.append(f"[{arm}]\n{diff}")
    return cur, "\n".join(notes)


def apply_to_targets(
    arms: Sequence[str], ctx: TransformContext
) -> tuple[dict[str, SourceDoc], dict[str, str]]:
    """Apply arms in order to every target page. Returns changed docs and per-doc diffs."""
    changed: dict[str, SourceDoc] = {}
    diffs: dict[str, str] = {}
    for doc_id, doc in ctx.docs.items():
        if doc.role != "target":
            continue
        cur, notes = doc, []
        for arm in arms:
            cur, diff = TRANSFORMS[arm].apply(cur, ctx)
            if diff:
                notes.append(f"[{arm}]\n{diff}")
        if cur is not doc:
            changed[doc_id] = cur
            diffs[doc_id] = "\n".join(notes)
    return changed, diffs
