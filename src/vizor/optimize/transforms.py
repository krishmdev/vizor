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
    _qvec: np.ndarray | None = field(default=None, repr=False)

    def query_vectors(self) -> np.ndarray:
        if self._qvec is None:
            self._qvec = self.embedder.encode([q.text for q in self.queries], kind="query")
        return self._qvec

    def doc_vector(self, doc: SourceDoc) -> np.ndarray:
        v = self.embedder.encode([doc.title + ". " + doc.body[:2000]], kind="passage")[0]
        return v / (np.linalg.norm(v) or 1.0)

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
_UNIT = (
    r"(?:kg|g|lb|lbs|psi|bar|mm|cm|m|km|km/h|mph|w|wh|nm|lumens?|%|percent|litres?|liters?|l|"
    r"inch|inches|in|hours?|minutes?|seconds?|°c|db|cycles|speed|x)"
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


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def grounding_violations(new: str, old: str) -> list[str]:
    """Numbers and names in `new` that do not appear in `old`. Numbers are compared without
    thousands separators. A capitalized word must appear with the same capitalization in `old`,
    except at the start of a sentence, where any case counts."""
    bad = sorted(_numbers(new) - _numbers(old))
    old_words = set(re.findall(r"[A-Za-z0-9&'’-]+", old))
    old_lower = {w.lower() for w in old_words}
    for sent in split_sentences(new.replace("\n", " ")):
        for m in _CAP.finditer(sent):
            w = m.group(0).rstrip("'’-")
            first = m.start() == len(sent) - len(sent.lstrip("\"'“([*-• "))
            if w in old_words or (first and w.lower() in old_lower):
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


def evidence_surface_llm(doc: SourceDoc, ctx: TransformContext) -> tuple[SourceDoc, str]:
    """GEO's statistics addition restricted to numbers already on the page: an LLM restates the
    page with its own numbers surfaced. A grounding guard rejects any output with a number or
    name that is not on the page, or that keeps fewer than 90% of the page's distinct numbers;
    a rejected page keeps its original text."""
    if ctx.llm is None:
        raise RuntimeError("evidence_surface_llm needs a rewriter LLM")
    user = (
        f"{EVIDENCE_PROMPT}\n\nReturn only the rewritten page text, with paragraphs separated "
        f"by blank lines and no commentary.\n\nPage:\n```\n{doc.body}\n```"
    )
    c = ctx.llm.complete(
        [{"role": "system", "content": GEO_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        seed=0,
        max_tokens=1500,
    )
    body = re.sub(r"^```[\w-]*\s*\n|\n?```\s*$", "", c.text.strip()).strip()
    bad = grounding_violations(body, doc.body + "\n" + doc.title) if body else ["<empty>"]
    # Surfacing must not become cutting: most of the page's own numbers have to survive.
    had, kept = _numbers(doc.body), _numbers(body) & _numbers(doc.body)
    if had and len(kept) < MIN_NUMBERS_KEPT * len(had):
        bad.append(f"<kept {len(kept)} of {len(had)} numbers>")
    ctx.events.append(
        {
            "transform": "evidence_surface_llm",
            "doc_id": doc.doc_id,
            "accepted": not bad,
            "violations": bad[:10],
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
