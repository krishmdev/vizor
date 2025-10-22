"""Vizor dashboard (Streamlit). Talks to the API at $VIZOR_API (default http://localhost:8000).

Four views: Overview, Answer inspector, Sandbox, Optimizer. Domain colors are fixed per entity
(project order: target first), validated for colour-vision deficiency, and every coloured mark
also carries its domain name, so colour is never the only cue.
"""

from __future__ import annotations

import difflib
import html
import os
import re

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("VIZOR_API", "http://localhost:8000").rstrip("/")

INK, INK_2, MUTED, RULE, PAPER, PANEL = (
    "#1d1c1a",
    "#52514e",
    "#8a8983",
    "#e4e2dc",
    "#fbfaf7",
    "#f3f1ec",
)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONT_BODY = "Charter, 'Iowan Old Style', 'Source Serif 4', Georgia, serif"
FONT_NUM = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace"

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --rule:{RULE}; --paper:{PAPER}; --panel:{PANEL}; }}
html, body, [class*="st-"], .stMarkdown, p, li, label {{ font-family:{FONT_BODY}; color:var(--ink); }}
h1, h2, h3 {{ font-family:{FONT_BODY}; letter-spacing:-0.01em; font-weight:600; }}
h1 {{ font-size:2.0rem !important; margin-bottom:0.2rem; }}
code, .num {{ font-family:{FONT_NUM}; font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top:3.6rem; max-width:1180px; }}
.kicker {{ font:600 0.72rem/1 {FONT_NUM}; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-bottom:0.35rem; }}
.lede {{ color:var(--ink2); font-size:1.02rem; max-width:62ch; margin:0 0 1.2rem 0; }}
.banner {{ border:1px solid #e9c46a; background:#fdf6e3; padding:0.6rem 0.9rem; border-radius:6px; font-size:0.92rem; margin-bottom:1rem; }}
.banner b {{ font-family:{FONT_NUM}; font-size:0.78rem; letter-spacing:0.08em; }}
.stat {{ border-top:2px solid var(--ink); padding-top:0.45rem; }}
.stat .v {{ font:600 1.7rem/1.1 {FONT_NUM}; }}
.stat .l {{ color:var(--ink2); font-size:0.85rem; }}
.src {{ display:grid; grid-template-columns: 2.2rem 1fr auto; gap:0.2rem 0.8rem; align-items:center;
        padding:0.55rem 0.7rem; border:1px solid var(--rule); border-left:4px solid var(--c); border-radius:6px;
        background:#fff; margin-bottom:0.45rem; }}
.src.ignored {{ background:var(--panel); border-left-style:dashed; }}
.src .idx {{ font:600 0.95rem {FONT_NUM}; color:var(--ink); }}
.src .dom {{ font:600 0.9rem {FONT_NUM}; }}
.src .ttl {{ color:var(--ink2); font-size:0.86rem; grid-column:2 / 4; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.src .share {{ font:600 0.95rem {FONT_NUM}; text-align:right; }}
.badge {{ display:inline-block; font:600 0.66rem/1 {FONT_NUM}; letter-spacing:0.1em; text-transform:uppercase;
          padding:0.28rem 0.45rem; border-radius:3px; margin-left:0.5rem; vertical-align:middle; }}
.badge.emphasized {{ background:var(--ink); color:#fff; }}
.badge.cited {{ border:1px solid var(--ink); color:var(--ink); }}
.badge.ignored {{ border:1px dashed var(--muted); color:var(--muted); }}
.badge.target {{ background:#e8f0fb; color:#1c5cab; }}
.answer {{ font-size:1.08rem; line-height:1.85; background:#fff; border:1px solid var(--rule); border-radius:8px; padding:1.1rem 1.3rem; }}
.sent {{ padding:0.1rem 0.15rem; border-radius:3px; box-decoration-break:clone; -webkit-box-decoration-break:clone;
         background:linear-gradient(transparent 62%, var(--tint) 62%); }}
.sent.none {{ background:none; color:var(--ink2); }}
.cite {{ font:600 0.72rem {FONT_NUM}; color:#fff; background:var(--c); border-radius:3px; padding:0.05rem 0.28rem; margin-left:0.12rem; vertical-align:0.12rem; }}
.bar {{ display:flex; gap:2px; height:14px; border-radius:4px; overflow:hidden; margin:0.3rem 0 0.2rem; }}
.bar span {{ display:block; height:100%; }}
.legend {{ display:flex; flex-wrap:wrap; gap:0.3rem 1rem; font:0.8rem {FONT_NUM}; color:var(--ink2); }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:0.35rem; vertical-align:-1px; }}
.note {{ color:var(--muted); font-size:0.84rem; }}
[data-testid="stSidebar"] {{ background:var(--panel); }}
@media (max-width: 640px) {{ .answer {{ font-size:1rem; padding:0.8rem; }} h1 {{ font-size:1.55rem !important; }} }}
</style>
"""


# ------------------------------------------------------------------ data access
@st.cache_data(ttl=30, show_spinner=False)
def get(path: str):
    r = httpx.get(API + path, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict):
    r = httpx.post(API + path, json=body, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def api_or_stop(path: str):
    try:
        return get(path)
    except httpx.HTTPError as e:
        st.error(
            f"Can't reach the Vizor API at `{API}` ({type(e).__name__}). Start it with "
            "`make serve` (or `docker compose up`), then reload this page."
        )
        st.stop()


@st.cache_resource
def egress_state() -> str:
    from vizor.egress import check_egress

    leaks = check_egress()
    return "blocked" if not leaks else "open: " + ", ".join(leaks)


def domain_colors(domains: dict[str, str]) -> dict[str, str]:
    order = [d for d, r in domains.items() if r == "target"] + [
        d for d, r in domains.items() if r != "target"
    ]
    return {d: SERIES[i % len(SERIES)] for i, d in enumerate(order)}


def tint(hex_color: str, alpha: float = 0.28) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def figure(height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_NUM, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#fff", bordercolor=RULE, font=dict(family=FONT_NUM, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=RULE, gridwidth=1, zeroline=False, linecolor=RULE, ticks=""
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=RULE, gridwidth=1, zeroline=False, linecolor=RULE, ticks=""
    )
    return fig


def header(kicker: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def fake_banner(is_fake: bool) -> None:
    if is_fake:
        st.markdown(
            '<div class="banner"><b>PIPELINE CHECK</b> &nbsp;These answers come from FakeLLM, a '
            "deterministic extractive stand-in. It shows the pipeline working; it says nothing about "
            "how a real model cites sources. Pick a real-model experiment in the sidebar for that.</div>",
            unsafe_allow_html=True,
        )


def stat(col, value: str, label: str) -> None:
    col.markdown(
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def pick_experiment() -> dict | None:
    exps = [e for e in api_or_stop("/experiments") if e["kind"] == "experiment"]
    if not exps:
        return None
    exps.sort(key=lambda e: (bool(e["is_fake_llm"]), e["location"] != "results", e["id"]))
    labels = [f"{e['id']}  ·  {'FakeLLM' if e['is_fake_llm'] else e['llm_model']}" for e in exps]
    i = st.sidebar.selectbox("Experiment", range(len(exps)), format_func=lambda k: labels[k])
    return exps[i]


def no_experiments() -> None:
    st.info(
        "No experiment results yet. Run `vizor demo` (keyless) or start one from the Sandbox view."
    )


# ------------------------------------------------------------------ views
def overview() -> None:
    header(
        "Overview",
        "Who the answer engine cites",
        "Baseline visibility of each domain across the tracked queries: how often it is retrieved into the "
        "prompt, how often the answer cites it, and how much of the answer it gets (position-adjusted word "
        "count share, PAWC, from Aggarwal et al. 2024).",
    )
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    e = api_or_stop(f"/experiments/{exp['id']}")
    m = e["manifest"]
    fake_banner(m.get("llm_is_fake", False))
    df = pd.DataFrame(e["domains"])
    roles = dict(zip(df.domain, df.role, strict=True))
    colors = domain_colors({d: roles[d] for d in df.domain})
    t = df[df.role == "target"].iloc[0]
    c = st.columns(4)
    stat(c[0], f"{t.retrieval_rate * 100:.0f}%", f"{t.domain} retrieved")
    stat(c[1], f"{t.citation_rate * 100:.0f}%", "of answers cite it")
    stat(c[2], f"{t.pawc_sov * 100:.1f}%", "PAWC share of answers")
    stat(c[3], f"{t.ignored_rate * 100:.0f}%", "ignored when retrieved")
    st.markdown(" ")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Share of the answer")
        st.markdown(
            '<p class="note">Two views of share. C-SoV counts citation markers; PAWC weights each '
            "cited sentence by its length and how early it appears.</p>",
            unsafe_allow_html=True,
        )
        d = df.sort_values("pawc_sov")
        fig = figure(60 + 46 * len(d))
        for metric, name, op in [
            ("c_sov", "Citation share of voice", 0.45),
            ("pawc_sov", "PAWC share", 1.0),
        ]:
            fig.add_bar(
                y=d.domain,
                x=d[metric] * 100,
                orientation="h",
                name=name,
                marker=dict(color=[colors[x] for x in d.domain], opacity=op, cornerradius=4),
                hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>",
            )
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_xaxes(ticksuffix="%", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("### Funnel per domain")
        tbl = pd.DataFrame(
            {
                "Domain": df.domain,
                "Retrieved": (df.retrieval_rate * 100).round(0),
                "Cited": (df.citation_rate * 100).round(0),
                "Cited if retrieved": (df.conversion * 100).round(0),
                "First cite (sentence)": df.first_cite_sentence.round(2),
                "Answer sentiment": df.answer_sentiment.round(2),
            }
        )
        st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.markdown(
        f'<p class="note">{m["n_queries"]} queries × {m["samples"]} samples · retrieval '
        f"<code>{html.escape(m['embedder_id'])}</code> + <code>{html.escape(m['reranker_id'])}</code> · answers "
        f"<code>{html.escape(m['llm_model'])}</code> · sentiment <code>{html.escape(m['sentiment_backend'])}</code></p>",
        unsafe_allow_html=True,
    )


def render_answer(a: dict, colors: dict[str, str]) -> None:
    by_pos = {s["position"]: s for s in a["sources"]}
    left, right = st.columns([3, 2], gap="large")
    with right:
        st.markdown("### Sources in the prompt")
        segs = "".join(
            f'<span title="[{s["position"]}] {s["domain"]} {s["pwc_share"]:.0%}" '
            f'style="width:{max(s["pwc_share"] * 100, 0.0):.2f}%;background:{colors.get(s["domain"], MUTED)}"></span>'
            for s in a["sources"]
            if s["pwc_share"] > 0
        )
        st.markdown(
            f'<div class="kicker">PAWC share of this answer</div><div class="bar">{segs}</div>',
            unsafe_allow_html=True,
        )
        cards = []
        for s in a["sources"]:
            c = colors.get(s["domain"], MUTED)
            tgt = '<span class="badge target">target</span>' if s["role"] == "target" else ""
            cards.append(
                f'<div class="src {s["label"]}" style="--c:{c}">'
                f'<span class="idx">[{s["position"]}]</span>'
                f'<span class="dom">{html.escape(s["domain"])}<span class="badge {s["label"]}">{s["label"]}</span>{tgt}</span>'
                f'<span class="share">{s["pwc_share"]:.0%}</span>'
                f'<span class="ttl" title="{html.escape(s["title"])}">{html.escape(s["title"] or s["url"])} · '
                f"score {s['final_score']:.2f} · {s['citations']} cite{'s' if s['citations'] != 1 else ''}</span></div>"
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
    with left:
        st.markdown("### Answer")
        parts = []
        for sent in a["sentences"]:
            cites = sent["citations"]
            text = html.escape(sent["text"])
            text = re.sub(r"\s*(\[\d+\])+\s*([.!?]?)$", r"\2", text)
            text = re.sub(r"\[\d+\]", "", text)
            if cites:
                c = colors.get(by_pos[cites[0]]["domain"], MUTED)
                chips = "".join(
                    f'<span class="cite" style="--c:{colors.get(by_pos[k]["domain"], MUTED)}" '
                    f'title="{html.escape(by_pos[k]["domain"])}">{k}</span>'
                    for k in cites
                )
                parts.append(f'<span class="sent" style="--tint:{tint(c)}">{text}</span>{chips} ')
            else:
                parts.append(f'<span class="sent none">{text}</span> ')
        st.markdown(f'<div class="answer">{"".join(parts)}</div>', unsafe_allow_html=True)
        if a["hallucinated_citations"]:
            st.warning(
                f"Citations to sources that weren't in the prompt (dropped): {a['hallucinated_citations']}"
            )
        with st.expander("Per-sentence attribution"):
            rows = [
                {
                    "pos": s["pos"],
                    "words": s["n_words"],
                    "decay": round(s["decay"], 3),
                    "cites": " ".join(f"[{k}] {by_pos[k]['domain']}" for k in s["citations"])
                    or "-",
                    "sentence": s["text"],
                }
                for s in a["sentences"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    present = {s["domain"] for s in a["sources"]}
    st.markdown(
        '<div class="legend">'
        + "".join(
            f'<span><i style="background:{c}"></i>{html.escape(d)}</span>'
            for d, c in colors.items()
            if d in present
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def inspector() -> None:
    header(
        "Answer inspector",
        "One answer, sentence by sentence",
        "Each sentence is underlined in the colour of the source it cites first; chips show every citation. "
        "Sources are labelled emphasized (largest PAWC share), cited, or ignored (shown to the model, never cited).",
    )
    proj = api_or_stop("/project")
    colors = domain_colors(proj["domains"])
    mode = st.sidebar.radio("Answers from", ["Stored experiment", "Live query"], horizontal=False)
    if mode == "Live query":
        fake_banner(proj["is_fake_llm"])
        with st.form("live"):
            q = st.text_input("Question", "best espresso machine for milk drinks")
            c1, c2 = st.columns(2)
            policy = c1.selectbox(
                "Retrieval policy",
                ["relevance", "reverse", "target_at:1", "target_at:5", "boost:0.05", "boost:0.1"],
            )
            arms = c2.multiselect(
                "Apply to target pages first",
                [
                    "metadata",
                    "faq_rewrite",
                    "jsonld_insert",
                    "internal_links",
                    "stats_surface",
                    "keyword_stuffing",
                ],
            )
            go_ = st.form_submit_button("Answer")
        if go_:
            with st.spinner("Retrieving, reranking and generating…"):
                try:
                    a = post("/answer", {"query": q, "policy": policy, "arms": arms})
                except (RuntimeError, httpx.HTTPError) as e:
                    st.error(f"Answer failed: {e}")
                    return
            render_answer(a, colors)
        return
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    fake_banner(exp["is_fake_llm"])
    e = api_or_stop(f"/experiments/{exp['id']}")
    arms = ["baseline"] + [d["arm"] for d in e["deltas"] if d["arm"] != "noop"]
    arm = st.sidebar.selectbox("Arm", arms)
    qtext = {q["query_id"]: q["query"] for q in proj["queries"]}
    idx = api_or_stop(f"/experiments/{exp['id']}/answers?arm={arm}")
    qids = sorted({r["query_id"] for r in idx})
    if not qids:
        st.info("No stored answers for this arm.")
        return
    c1, c2 = st.columns([4, 1])
    qid = c1.selectbox("Query", qids, format_func=lambda k: f"{qtext.get(k, k)}  ({k})")
    samples = sorted({r["sample"] for r in idx if r["query_id"] == qid})
    sample = c2.selectbox("Sample", samples)
    a = api_or_stop(f"/experiments/{exp['id']}/answers/{qid}/{sample}?arm={arm}")
    render_answer(a, colors)


def _interval_band(fig, x, lo, hi, color, name):
    fig.add_scatter(
        x=list(x) + list(x)[::-1],
        y=list(hi) + list(lo)[::-1],
        fill="toself",
        fillcolor=tint(color, 0.16),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name=name,
    )


def sandbox() -> None:
    header(
        "Sandbox",
        "What changes the target's share",
        "Each arm edits the target site's pages (or the engine's retrieval) and re-runs the same queries with the "
        "same seeds. Intervals are paired-bootstrap 95% CIs over queries; p-values are Wilcoxon with Holm correction. "
        "The A/A row re-samples unchanged prompts: read every other arm against it.",
    )
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    fake_banner(exp["is_fake_llm"])
    e = api_or_stop(f"/experiments/{exp['id']}")
    d = pd.DataFrame(e["deltas"])
    aa = d[d.arm == "aa_resample"]
    d = d.iloc[::-1]
    st.markdown("### ΔPAWC share of the target, percentage points")
    fig = figure(70 + 38 * len(d))
    if len(aa):
        fig.add_vrect(
            x0=float(aa.d_pwc_lo.iloc[0]),
            x1=float(aa.d_pwc_hi.iloc[0]),
            fillcolor=PANEL,
            line_width=0,
            annotation_text="A/A noise band",
            annotation_position="top left",
            annotation_font=dict(size=11, color=MUTED),
        )
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    sig = (d.d_pwc_lo > 0) | (d.d_pwc_hi < 0)
    col = [SERIES[0] if s else MUTED for s in sig]
    fig.add_scatter(
        x=d.d_pwc_pp,
        y=d.arm,
        mode="markers",
        marker=dict(size=10, color=col, line=dict(color="#fff", width=2)),
        error_x=dict(
            type="data",
            symmetric=False,
            array=d.d_pwc_hi - d.d_pwc_pp,
            arrayminus=d.d_pwc_pp - d.d_pwc_lo,
            color=INK_2,
            thickness=1.5,
            width=0,
        ),
        customdata=d[["d_pwc_lo", "d_pwc_hi", "p_holm", "n_queries"]].to_numpy(),
        hovertemplate="%{y}<br>Δ %{x:+.2f} pp [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]"
        "<br>Holm p %{customdata[2]:.3f} · n=%{customdata[3]}<extra></extra>",
        showlegend=False,
    )
    fig.update_xaxes(ticksuffix=" pp")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<p class="note">Blue: CI excludes zero. Gray: CI includes zero.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Position sweep")
        st.markdown(
            '<p class="note">Same pages, target forced into slot 1–5 of the prompt.</p>',
            unsafe_allow_html=True,
        )
        p = pd.DataFrame(e["position_sweep"])
        if len(p):
            fig = figure(300)
            _interval_band(fig, p.position, p.pwc_lo, p.pwc_hi, SERIES[0], "ci")
            fig.add_scatter(
                x=p.position,
                y=p.pwc_pct,
                mode="lines+markers",
                line=dict(color=SERIES[0], width=2),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate="slot %{x}: %{y:.1f}%<extra></extra>",
            )
            fig.update_xaxes(dtick=1, title="slot of the target in the prompt")
            fig.update_yaxes(ticksuffix="%", rangemode="tozero", title="PAWC share")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("### Boost sweep")
        st.markdown(
            '<p class="note">w added to the target pages\' final retrieval score (0–1 scale).</p>',
            unsafe_allow_html=True,
        )
        b = pd.DataFrame(e["boost_sweep"])
        if len(b):
            fig = figure(300)
            _interval_band(fig, b.boost, b.pwc_lo, b.pwc_hi, SERIES[0], "ci")
            fig.add_scatter(
                x=b.boost,
                y=b.pwc_pct,
                mode="lines+markers",
                line=dict(color=SERIES[0], width=2),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate="w=%{x}: %{y:.1f}%<extra></extra>",
            )
            fig.update_xaxes(title="boost w")
            fig.update_yaxes(ticksuffix="%", rangemode="tozero", title="PAWC share")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Page diffs")
    diffs = api_or_stop(f"/experiments/{exp['id']}/diffs")
    if diffs:
        c1, c2, c3 = st.columns([2, 1, 3])
        arm = c1.selectbox("Arm", list(diffs))
        fold = c2.selectbox(
            "Fold",
            list(diffs[arm]),
            help="Arms that read the tracked queries are built from "
            "the other fold's queries and scored on this fold's.",
        )
        pages = diffs[arm][fold]
        doc = c3.selectbox("Page", list(pages), format_func=lambda k: pages[k]["url"])
        rec = pages[doc]
        diff = difflib.unified_diff(
            rec["before"].splitlines(),
            rec.get("after", "").splitlines(),
            "before",
            "after",
            lineterm="",
            n=1,
        )
        st.code("\n".join(diff) or "(no change)", language="diff")
    else:
        st.markdown('<p class="note">No page arms in this experiment.</p>', unsafe_allow_html=True)

    with st.expander("Table view"):
        st.dataframe(d.iloc[::-1], hide_index=True, use_container_width=True)
    with st.expander("Run a new sandbox experiment"):
        with st.form("job"):
            nq = st.slider("Queries", 4, 40, 8)
            ns = st.slider("Samples per query", 1, 5, 2)
            ok = st.form_submit_button("Start")
        if ok:
            try:
                job = post("/sandbox", {"queries": nq, "samples": ns})
                st.success(f"Started {job['job_id']}. It appears in the experiment list when done.")
            except (RuntimeError, httpx.HTTPError) as err:
                st.error(str(err))


def optimizer() -> None:
    header(
        "Optimizer",
        "Learning which fix to apply",
        "A contextual bandit picks one page optimization per query context and observes the measured change in PAWC "
        "share. Evaluated by offline replay over the sandbox's reward table; regret is measured against the best arm "
        "for each query.",
    )
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    fake_banner(exp["is_fake_llm"])
    e = api_or_stop(f"/experiments/{exp['id']}")
    curve = pd.DataFrame(e["bandit_curve"])
    if not len(curve):
        st.info("This experiment has no bandit replay.")
        return
    summ = pd.DataFrame(e["bandit_summary"])
    order = ["linucb(a=0.1)", "lints(v=0.1)", "eps-greedy(0.1)", "best-fixed-arm", "random"]
    shown = [p for p in order if p in set(curve.policy)]
    colors = {p: SERIES[i] for i, p in enumerate(shown)}
    fig = figure(380)
    for p in shown:
        c = curve[curve.policy == p]
        _interval_band(fig, c.t, c.lo, c.hi, colors[p], p)
        fig.add_scatter(
            x=c.t,
            y=c.cum_regret,
            mode="lines",
            name=p,
            line=dict(color=colors[p], width=2),
            hovertemplate=p + "<br>t=%{x}: %{y:.2f}<extra></extra>",
        )
        last = c.iloc[-1]
        fig.add_annotation(
            x=last.t,
            y=last.cum_regret,
            text=p,
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=INK_2),
        )
    fig.update_xaxes(title="round")
    fig.update_yaxes(title="cumulative regret (PAWC share)", rangemode="tozero")
    fig.update_layout(margin=dict(r=110))
    st.markdown("### Cumulative regret, mean of replay runs with 95% band")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(summ, hide_index=True, use_container_width=True)
    traj = pd.DataFrame(e["trajectory"])
    if len(traj):
        st.markdown("### Greedy loop on held-out queries")
        st.markdown(
            '<p class="note">The bandit, trained on half the queries, proposes arms in order; an arm is kept '
            "only if the lower 95% bound of its held-out ΔPAWC is above zero.</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(traj, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Vizor", page_icon=None, layout="wide", initial_sidebar_state="expanded"
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="kicker">Vizor</div><div class="note">API <code>{html.escape(API)}</code></div>',
        unsafe_allow_html=True,
    )
    if os.environ.get("VIZOR_EGRESS_CANARY") == "1":
        st.sidebar.markdown(
            f'<div class="note">egress: {egress_state()}</div>', unsafe_allow_html=True
        )
    pages = [
        st.Page(overview, title="Overview", url_path="overview", default=True),
        st.Page(inspector, title="Answer inspector", url_path="inspector"),
        st.Page(sandbox, title="Sandbox", url_path="sandbox"),
        st.Page(optimizer, title="Optimizer", url_path="optimizer"),
    ]
    st.navigation(pages).run()


main()
