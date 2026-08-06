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

from vizor.optimize.sensitivity import reachable

API = os.environ.get("VIZOR_API", "http://localhost:8000").rstrip("/")

INK, INK_2, MUTED, RULE, PAPER, PANEL = (
    "#1d1c1a",
    "#52514e",
    "#6b6a65",
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
html, body, .stMarkdown, p, li, label, input, textarea {{ font-family:{FONT_BODY}; color:var(--ink); }}
[data-testid^="stBaseButton-primary"] p {{ color:#fff !important; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family:"Material Symbols Rounded" !important; }}
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
.badge.fake {{ background:#fdf6e3; color:#6b4e00; border:1px solid #e9c46a; }}
.runline {{ color:var(--ink2); font-size:0.9rem; margin:-0.4rem 0 1rem 0; }}
.psg {{ margin:0.1rem 0 0.8rem 2.2rem; padding:0; list-style:none; font-size:0.9rem; line-height:1.5; }}
.psg li {{ padding:0.15rem 0.4rem; border-left:3px solid var(--rule); margin-bottom:0.2rem; }}
.psg li.new {{ border-left-color:#b45309; background:#fef3c7; }}
.answer {{ font-size:1.08rem; line-height:1.85; background:#fff; border:1px solid var(--rule); border-radius:8px; padding:1.1rem 1.3rem; }}
.sent {{ padding:0.1rem 0.15rem; border-radius:3px; box-decoration-break:clone; -webkit-box-decoration-break:clone;
         background:linear-gradient(transparent 62%, var(--tint) 62%); }}
.sent.none {{ background:none; color:var(--ink2); }}
.cite {{ font:600 0.72rem {FONT_NUM}; color:var(--ink); background:#fff; border:1.5px solid var(--c); border-radius:3px; padding:0 0.26rem; margin-left:0.12rem; vertical-align:0.12rem; }}
code {{ color:var(--ink2); background:var(--panel); border-radius:3px; padding:0 0.25rem; }}
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
    committed = [e for e in exps if e["location"] == "results"]
    local = [e for e in exps if e["location"] != "results"]
    groups = {"Committed results": committed, "Local runs": local}
    names = [g for g, v in groups.items() if v]
    group = st.sidebar.radio("Runs", names, horizontal=True) if len(names) > 1 else names[0]
    choices = sorted(groups[group], key=lambda e: (bool(e["is_fake_llm"]), e["id"]))
    labels = [
        e["id"] + (" (FakeLLM)" if e["is_fake_llm"] and "fakellm" not in e["id"].lower() else "")
        for e in choices
    ]
    i = st.sidebar.selectbox("Experiment", range(len(choices)), format_func=lambda k: labels[k])
    return choices[i]


def run_line(exp: dict) -> None:
    """Which run the page shows, directly under the heading."""
    model = "FakeLLM" if exp["is_fake_llm"] else html.escape(exp["llm_model"] or "")
    badge = '<span class="badge fake">FakeLLM · pipeline check</span>' if exp["is_fake_llm"] else ""
    st.markdown(
        f'<p class="runline"><code>{html.escape(exp["id"])}</code> · {model} · '
        f"{exp['n_queries']} queries × {exp['samples']} samples{badge}</p>",
        unsafe_allow_html=True,
    )


def titled(text: str, exp: dict) -> str:
    return text + (" · FakeLLM pipeline check" if exp["is_fake_llm"] else "")


def no_experiments() -> None:
    st.info("No experiment results yet. Run `vizor demo` (keyless), or start a small one here.")
    job_form()


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
    run_line(exp)
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
        st.markdown(f"### {titled('Share of the answer', exp)}")
        st.markdown(
            '<p class="note">Two views of share. Pooled marker share counts citation markers across all answers; PAWC weights each '
            "cited sentence by its length and how early it appears.</p>",
            unsafe_allow_html=True,
        )
        d = df.sort_values("pawc_sov")
        ylab = [
            f"<b>{x} (target)</b>" if r == "target" else x
            for x, r in zip(d.domain, d.role, strict=True)
        ]
        fig = figure(80 + 46 * len(d))
        for metric, name, op in [
            ("c_sov", "Pooled marker share", 0.45),
            ("pawc_sov", "PAWC share", 1.0),
        ]:
            fig.add_bar(
                y=ylab,
                x=d[metric] * 100,
                orientation="h",
                name=name,
                showlegend=False,
                marker=dict(color=[colors[x] for x in d.domain], opacity=op, cornerradius=4),
                hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>",
            )
        # legend swatches for the two encodings (the bars themselves are coloured by domain)
        for name, op in [("Citation share of voice (light)", 0.45), ("PAWC share (solid)", 1.0)]:
            fig.add_bar(y=[None], x=[None], name=name, marker=dict(color=INK_2, opacity=op))
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_xaxes(ticksuffix="%", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("### Funnel per domain")
        tbl = pd.DataFrame(
            {
                "Domain": [
                    f"{x} ★" if r == "target" else x
                    for x, r in zip(df.domain, df.role, strict=True)
                ],
                "Retr.": df.retrieval_rate * 100,
                "Cited": df.citation_rate * 100,
                "Conv.": df.conversion * 100,
                "PAWC": df.pawc_sov * 100,
                "Pooled marker share": df.c_sov * 100,
                "1st cite": df.first_cite_sentence,
                "Sent.": df.answer_sentiment,
            }
        )

        def pct(label: str, helptext: str):
            return st.column_config.NumberColumn(label, help=helptext, format="%.0f%%")

        st.dataframe(
            tbl,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Domain": st.column_config.TextColumn("Domain", help="★ marks the target site"),
                "Retr.": pct("Retr.", "Share of answers whose prompt included this domain"),
                "Cited": pct("Cited", "Share of answers citing this domain at least once"),
                "Conv.": pct("Conv.", "Cited when retrieved"),
                "PAWC": st.column_config.NumberColumn(
                    "PAWC", help="Mean PAWC share of the answer", format="%.1f%%"
                ),
                "C-SoV": st.column_config.NumberColumn(
                    "C-SoV", help="Share of all citation markers", format="%.1f%%"
                ),
                "1st cite": st.column_config.NumberColumn(
                    "1st cite", help="Mean sentence of first citation", format="%.2f"
                ),
                "Sent.": st.column_config.NumberColumn(
                    "Sent.", help="PAWC-weighted sentiment of citing sentences", format="%+.2f"
                ),
            },
        )
        st.markdown(
            '<p class="note">★ target site. Hover a header for its definition.</p>',
            unsafe_allow_html=True,
        )
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
        present = []
        for s_ in a["sources"]:
            if s_["domain"] not in present:
                present.append(s_["domain"])
        legend = "".join(
            f'<span><i style="background:{colors.get(d_, MUTED)}"></i>{html.escape(d_)}</span>'
            for d_ in present
        )
        st.markdown(
            f'<div class="kicker">PAWC share of this answer</div><div class="bar" role="img" '
            f'aria-label="PAWC share by source">{segs}</div><div class="legend">{legend}</div>',
            unsafe_allow_html=True,
        )
        if not any(s_["role"] == "target" for s_ in a["sources"]):
            st.info("The target site wasn't retrieved into this prompt, so it can't be cited here.")
        shares = [round(s_["pwc_share"], 4) for s_ in a["sources"] if s_["pwc_share"] > 0]
        tied = {x for x in shares if shares.count(x) > 1}
        cards = []
        for s in a["sources"]:
            c = colors.get(s["domain"], MUTED)
            tgt = '<span class="badge target">target</span>' if s["role"] == "target" else ""
            cards.append(
                f'<div class="src {s["label"]}" style="--c:{c}">'
                f'<span class="idx">[{s["position"]}]</span>'
                f'<span class="dom">{html.escape(s["domain"])}<span class="badge {s["label"]}">{s["label"]}</span>{tgt}</span>'
                f'<span class="share">{s["pwc_share"]:.0%}{" (tie)" if round(s["pwc_share"], 4) in tied else ""}</span>'
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
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            if cites:
                c = colors.get(by_pos[cites[0]]["domain"], MUTED)
                chips = "".join(
                    f'<span class="cite" style="--c:{colors.get(by_pos[k]["domain"], MUTED)}" '
                    f'title="{html.escape(by_pos[k]["domain"])}" '
                    f'aria-label="cites source {k}, {html.escape(by_pos[k]["domain"])}">{k}</span>'
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


def inspector() -> None:
    header(
        "Answer inspector",
        "One answer, sentence by sentence",
        "Each sentence is underlined in the colour of the source it cites first; chips show every citation. "
        "Sources are labelled emphasized (largest PAWC share), cited, or ignored (shown to the model, never cited).",
    )
    with st.spinner("Loading retrieval models…"):
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
            go_ = st.form_submit_button("Answer", type="primary")
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
    run_line(exp)
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
        mode="lines",
        fillcolor=tint(color, 0.16),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name=name,
    )


def forest(d: pd.DataFrame, sens: dict, exp: dict) -> None:
    """Per-arm delta on the run's primary metric with descriptive CIs, grouped controls / page
    arms / content-only arms / engine arms, with the
    Holm verdict and n written beside each row so significance is never colour-only."""
    d = d[d.arm != "noop"].copy()
    primary = d["primary"].iloc[0] if "primary" in d and len(d) else "imp_pwc"
    col = {"imp_pwc": "d_pwc", "c_share": "d_csov", "mentioned": "d_mention"}[primary]
    metric = {"imp_pwc": "PAWC share", "c_share": "C-SoV", "mentioned": "named rate"}[primary]
    d["_m"], d["_lo"], d["_hi"] = d[f"{col}_pp"], d[f"{col}_lo"], d[f"{col}_hi"]
    group = {"aa": "Control", "doc": "Page arm", "engine": "Engine arm"}
    d["group"] = d.kind.map(group).fillna("Engine arm")
    if "mode" in d:
        d.loc[d["mode"] == "content", "group"] = "Content-only arm"
    order = {"Control": 0, "Page arm": 1, "Content-only arm": 2, "Engine arm": 3}
    d = d.sort_values(by=["group", "_m"], key=lambda c: c.map(order) if c.name == "group" else c)
    # short row labels so the chart fits a phone; the group is in the hover and the row order
    d["label"] = d.arm.str.replace("engine:", "", regex=False)
    labels = list(d.label)[::-1]  # plotly draws the first category at the bottom
    sig = d.get("significant", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    st.markdown(f"### {titled(f'Δ{metric} of the target', exp)}")
    fig = figure(90 + 40 * len(d))
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    page_ok = sens.get("page_arms_testable", True)
    family = int(sens.get("arm_holm_family") or len(d[d.kind != "aa"]))
    for kind, key in (("engine", "engine_arm_pp"), ("doc", "page_arm_pp")):
        m = sens.get(key)
        rows = d[d.kind == kind]
        units_col = rows["n_units"] if "n_units" in rows else rows["n_queries"]
        rows = rows[[reachable(int(u), family) for u in units_col.fillna(rows.n_queries)]]
        if m and len(rows):
            for sign in (-1, 1):
                fig.add_scatter(
                    x=[sign * m] * len(rows),
                    y=rows.label,
                    mode="markers",
                    showlegend=False,
                    marker=dict(symbol="line-ns", size=16, line=dict(color=MUTED, width=1.5)),
                    hovertemplate=f"MDE ±{m:.1f} pp (80% power)<extra></extra>",
                )
    for mask, symbol, color, name in (
        (sig, "diamond", SERIES[0], "Holm p < 0.05"),
        (~sig & (d.kind != "aa"), "circle", MUTED, "not significant"),
        (d.kind == "aa", "circle-open", INK_2, "A/A control (noise)"),
    ):
        rows = d[mask]
        if not len(rows):
            continue
        fig.add_scatter(
            x=rows._m,
            y=rows.label,
            mode="markers",
            name=name,
            marker=dict(
                size=11,
                symbol=symbol,
                color=color,
                line=dict(color=color if "open" in symbol else "#fff", width=2),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=rows._hi - rows._m,
                arrayminus=rows._m - rows._lo,
                color=INK_2,
                thickness=1.5,
                width=0,
            ),
            customdata=rows[["_lo", "_hi", "p_holm", "n_queries", "group"]].to_numpy(),
            hovertemplate="%{customdata[4]} · %{y}<br>Δ %{x:+.2f} pp [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]"
            "<br>Holm p %{customdata[2]:.3f} · n=%{customdata[3]}<extra></extra>",
        )
    for x in d.itertuples():
        units = int(getattr(x, "n_units", x.n_queries) or x.n_queries)
        if x.kind == "aa":
            txt = f"control n={int(x.n_queries)}"
        elif x.kind == "doc" and not reachable(units, family):
            txt = f"untestable k={units}"
        else:
            txt = f"p={x.p_holm:.2f}{'*' if bool(getattr(x, 'significant', False)) else ''} n={int(x.n_queries)}"
            if x.kind == "doc":
                txt += f"/{units}"
        fig.add_annotation(
            x=1.0,
            xref="paper",
            xanchor="left",
            xshift=10,
            y=x.label,
            text=txt,
            showarrow=False,
            font=dict(size=11, color=INK_2),
        )
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    fig.update_xaxes(ticksuffix=" pp", title=f"Δ{metric}, pp (95% bootstrap CI)")
    fig.update_layout(margin=dict(r=120, l=4), legend=dict(y=1.08))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tested = d[d.kind != "aa"]
    msg = (
        "No arm is distinguishable from zero at this sample size (Holm-adjusted Wilcoxon, α = 0.05)."
        if not sig.any()
        else f"{int(sig.sum())} arm(s) pass the Holm rule, marked ◆ and *."
    )
    msg = (
        "Rows, top to bottom: the A/A control, page arms, content-only arms, engine arms; k is the number of "
        "independent edited-page units. " + msg
    )
    extra = []
    if sens.get("engine_arm_pp"):
        extra.append(
            f"grey ticks = ±MDE (80% power): {sens['engine_arm_pp']:.1f} pp for engine arms"
        )
    if sens.get("n_page_units") is not None:
        extra.append(
            f"page arms tested on {sens.get('n_page_units_min', sens['n_page_units'])}–{sens['n_page_units']} page units"
            + ("" if page_ok else ", too few for any Holm-significant result")
        )
    if len(tested):
        st.markdown(
            f'<p class="note">{msg} '
            + "; ".join(extra).capitalize()[:1]
            + "; ".join(extra)[1:]
            + ".</p>",
            unsafe_allow_html=True,
        )


def sweep_chart(df: pd.DataFrame, xcol: str, dcol: str, xtitle: str, xname: str) -> None:
    """Measured points only (no lines through unmeasured slots), CI whiskers, and the Holm verdict
    of each comparison against the reference point, marked by shape and * as well as colour."""
    sig = df.get("significant", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    sig.iloc[0] = False
    fig = figure(300)
    for mask, color, symbol, name in (
        (~sig, MUTED, "circle", "not significant vs reference"),
        (sig, SERIES[0], "diamond", "Holm p < 0.05 vs reference"),
    ):
        rows = df[mask]
        if not len(rows):
            continue
        fig.add_scatter(
            x=rows[xcol],
            y=rows.pwc_pct,
            mode="markers+text",
            name=name,
            text=["*" if s else "" for s in sig[mask]],
            textposition="top center",
            textfont=dict(size=16, color=SERIES[0]),
            marker=dict(size=11, color=color, symbol=symbol),
            error_y=dict(
                type="data",
                symmetric=False,
                array=rows.pwc_hi - rows.pwc_pct,
                arrayminus=rows.pwc_pct - rows.pwc_lo,
                color=INK_2,
                thickness=1.5,
                width=6,
            ),
            customdata=rows[[dcol, "d_lo", "d_hi"]].to_numpy(),
            hovertemplate=f"{xname}=%{{x}}: %{{y:.1f}}%<br>Δ vs reference %{{customdata[0]:+.1f}} pp "
            "[%{customdata[1]:+.1f}, %{customdata[2]:+.1f}]<extra></extra>",
        )
    fig.update_xaxes(title=xtitle, tickvals=list(df[xcol]))
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", title="PAWC share")
    fig.update_layout(legend=dict(y=1.12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def sandbox() -> None:
    header(
        "Sandbox",
        "What changes the target's share",
        "Each arm edits the target site's pages (or the engine's retrieval) and re-runs the same queries with the "
        "same seeds. An arm counts as an effect only if its Holm-adjusted Wilcoxon p is below 0.05; the intervals are "
        "descriptive paired-bootstrap 95% CIs. "
        "The A/A row re-samples unchanged prompts: read every other arm against it.",
    )
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    run_line(exp)
    fake_banner(exp["is_fake_llm"])
    e = api_or_stop(f"/experiments/{exp['id']}")
    d = pd.DataFrame(e["deltas"])
    sens = e["manifest"].get("sensitivity") or {}
    forest(d, sens, exp)

    c1, c2 = st.columns(2, gap="large")
    p = pd.DataFrame(e["position_sweep"])
    b = pd.DataFrame(e["boost_sweep"])
    with c1:
        st.markdown(f"### {titled('Position sweep: vs slot 1', exp)}")
        st.markdown(
            '<p class="note">Same pages; the target is forced into each measured slot of the prompt.</p>',
            unsafe_allow_html=True,
        )
        if len(p):
            sweep_chart(
                p, "position", "d_pwc_vs_first_pp", "slot of the target in the prompt", "slot"
            )
    with c2:
        st.markdown(f"### {titled('Boost sweep: vs w=0', exp)}")
        st.markdown(
            '<p class="note">w is added to the target pages\' final retrieval score (0–1 scale).</p>',
            unsafe_allow_html=True,
        )
        if len(b):
            sweep_chart(b, "boost", "d_pwc_pp", "boost w", "w")
    with st.expander("Sweep tables"):
        if len(p):
            st.dataframe(p, hide_index=True, use_container_width=True)
        if len(b):
            st.dataframe(b, hide_index=True, use_container_width=True)

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
        st.code("\n".join(diff) or "(no change)", language="diff", wrap_lines=True)
    else:
        st.markdown('<p class="note">No page arms in this experiment.</p>', unsafe_allow_html=True)

    with st.expander("Table view"):
        st.dataframe(d.iloc[::-1], hide_index=True, use_container_width=True)
    with st.expander("Run a new sandbox experiment"):
        job_form()


def job_form() -> None:
    with st.form("job"):
        nq = st.slider("Queries", 4, 40, 8)
        ns = st.slider("Samples per query", 1, 5, 2)
        ok = st.form_submit_button("Start", type="primary")
    if ok:
        try:
            job = post("/sandbox", {"queries": nq, "samples": ns})
            st.success(f"Started {job['job_id']}. It appears under Local runs when done.")
        except (RuntimeError, httpx.HTTPError) as err:
            st.error(str(err))


def optimizer() -> None:
    header(
        "Optimizer",
        "Choosing which fix to apply",
        "A contextual bandit picks one page optimization per query context and observes the measured change in PAWC "
        "share. The replay curves are in-sample (rewards come from the table the policies learn from); the held-out "
        "table below is the real test of whether a learned choice carries over to new queries.",
    )
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    run_line(exp)
    fake_banner(exp["is_fake_llm"])
    e = api_or_stop(f"/experiments/{exp['id']}")
    curve = pd.DataFrame(e["bandit_curve"])
    if not len(curve):
        st.info("This experiment has no bandit replay.")
        return
    summ = pd.DataFrame(e["bandit_summary"])
    if "eval" not in summ:
        summ["eval"] = "replay"
    replay_s, held = summ[summ["eval"] == "replay"], summ[summ["eval"] == "heldout"]

    def table(df: pd.DataFrame, regret_label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Policy": df.policy,
                regret_label: [
                    f"{r * 100:.1f} ± {c * 100:.1f}" if c == c else f"{r * 100:.1f}"
                    for r, c in zip(df.final_regret, df.final_regret_ci, strict=True)
                ],
                "Mean reward (pp)": df.mean_reward_pp.round(2),
                "Most chosen arm": [
                    f"{a} ({sh:.0%})" if isinstance(a, str) and a else ""
                    for a, sh in zip(df.top_arm, df.top_arm_share, strict=True)
                ],
            }
        )

    if len(held):
        st.markdown(f"### {titled('Held-out check', exp)}")
        st.markdown(
            '<p class="note">Fit on one half of the queries, freeze, score the other half. This is the test '
            "of whether a learned choice carries over. Regret is summed over held-out queries, in pp of PAWC share.</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            table(held, "Held-out regret (pp, ±95% CI)"), hide_index=True, use_container_width=True
        )

    # the best of the configured LinUCB alphas, plus one of each other policy family
    linucb = replay_s[replay_s.policy.str.match(r"linucb\(a=")]
    best_lin = linucb.sort_values("final_regret").policy.iloc[0] if len(linucb) else None
    wanted = [best_lin, "lints(v=0.1)", "eps-greedy(0.1)", "linucb-bias-only(a=0.1)", "random"]
    shown = [x for x in wanted if x and x in set(curve.policy)]
    reference = [x for x in curve.policy.unique() if "hindsight" in x]
    colors = {x: SERIES[k] for k, x in enumerate(shown)}
    fig = figure(400)
    t_max = int(curve.t.max())
    ends = []
    for pol in shown + reference:
        c = curve[curve.policy == pol]
        ref = pol in reference
        color = MUTED if ref else colors[pol]
        if not ref:
            _interval_band(fig, c.t, c.lo * 100, c.hi * 100, color, pol)
        fig.add_scatter(
            x=c.t,
            y=c.cum_regret * 100,
            mode="lines",
            name=pol,
            showlegend=False,
            line=dict(color=color, width=1.5 if ref else 2),
            hovertemplate=pol + "<br>round %{x}: %{y:.0f} pp<extra></extra>",
        )
        ends.append([float(c.cum_regret.iloc[-1] * 100), pol, color])
    # nudge end labels apart so they don't overlap
    ends.sort()
    gap = max(1.0, max(e_[0] for e_ in ends) * 0.045)
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, pol, color in ends:
        label = pol
        fig.add_annotation(
            x=t_max,
            y=y,
            text=label,
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=color if color != MUTED else INK_2),
        )
    fig.update_xaxes(title="round", range=[0, t_max])
    fig.update_yaxes(title="cumulative regret (pp of PAWC share)", rangemode="tozero")
    fig.update_layout(margin=dict(r=210))
    fig.update_annotations(selector=dict(xanchor="left"), captureevents=False)
    st.markdown(f"### {titled('Replay regret (in-sample)', exp)}")
    st.markdown(
        f'<p class="note">Mean of replay runs with 95% band. LinUCB is shown at its best configured alpha '
        f"({html.escape(best_lin or '-')}); grey is the best single arm chosen in hindsight, a reference "
        "line rather than a policy.</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("Replay table and curve data"):
        st.dataframe(
            table(replay_s, "Cumulative regret (pp, ±95% CI)"),
            hide_index=True,
            use_container_width=True,
        )
        st.dataframe(curve, hide_index=True, use_container_width=True)
    traj = pd.DataFrame(e["trajectory"])
    if len(traj):
        st.markdown("### Greedy loop on held-out queries")
        st.markdown(
            '<p class="note">The bandit, trained on half the queries, proposes arms in order; an arm is kept '
            "only if the lower 95% bound of its held-out ΔPAWC is above zero.</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Step": traj.step,
                    "Proposed": traj.proposed,
                    "Predicted (pp)": traj.predicted_reward_pp.round(2),
                    "Held-out ΔPAWC (pp)": [
                        f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                        for m, lo, hi in zip(traj.d_pwc_pp, traj.d_lo, traj.d_hi, strict=True)
                    ],
                    "Kept": ["✓" if k else "–" for k in traj.kept],
                    "Applied so far": traj.applied,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def _passages_html(side: dict, colors: dict[str, str], other: dict) -> str:
    """Rendered passages per source; lines the other side doesn't show are marked."""
    theirs = {ln for p in other.get("passages", []) for ln in p["lines"]}
    out = []
    for p in side["passages"]:
        c = colors.get(p["domain"], MUTED)
        lines = "".join(
            f'<li class="{"" if ln in theirs else "new"}">{html.escape(ln)}</li>'
            for ln in p["lines"]
        )
        out.append(
            f'<div class="src" style="--c:{c}"><span class="idx">[{p["position"]}]</span>'
            f'<span class="dom">{html.escape(p["domain"])}</span>'
            f'<span class="ttl">{html.escape(p["title"])}</span></div><ul class="psg">{lines}</ul>'
        )
    return "".join(out)


def side_by_side() -> None:
    header(
        "Side by side",
        "Baseline and arm for the same query and sample",
        "Each column shows the passages the model saw for every source (lines only one side shows "
        "are highlighted) and, if the arm was sampled, its answer. Below, the reference answer's "
        "citation sites with the target's teacher-forced probability under each prompt.",
    )
    proj = api_or_stop("/project")
    colors = domain_colors(proj["domains"])
    exp = pick_experiment()
    if exp is None:
        return no_experiments()
    run_line(exp)
    sets = api_or_stop(f"/experiments/{exp['id']}/sets")
    others = [s for s in sets if s != "baseline"]
    if not others:
        st.info("This run has no arm to compare with the baseline.")
        return
    arm = st.sidebar.selectbox("Arm", others)
    ref = st.sidebar.selectbox(
        "Against", sets, index=sets.index("baseline") if "baseline" in sets else 0
    )
    qtext = {q["query_id"]: q["query"] for q in proj["queries"]}
    idx = api_or_stop(f"/experiments/{exp['id']}/answers?arm=baseline")
    qids = sorted({r["query_id"] for r in idx})
    c1, c2 = st.columns([4, 1])
    qid = c1.selectbox("Query", qids, format_func=lambda k: f"{qtext.get(k, k)}  ({k})")
    sample = c2.selectbox("Sample", sorted({r["sample"] for r in idx if r["query_id"] == qid}))
    d = api_or_stop(f"/experiments/{exp['id']}/compare/{qid}/{sample}?arm={arm}&ref={ref}")
    left, right = st.columns(2, gap="large")
    for col, side, other in (
        (left, d["sides"][0], d["sides"][1]),
        (right, d["sides"][1], d["sides"][0]),
    ):
        with col:
            ap = d["ap"].get(side["set"])
            st.markdown(f"### `{side['set']}`" + (f" · AP {ap:.1%}" if ap is not None else ""))
            if side.get("answer"):
                st.markdown(
                    f'<div class="answer">{html.escape(side["answer"]["text"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Prompt only: no answer was sampled for this set.")
            st.markdown(_passages_html(side, colors, other), unsafe_allow_html=True)
    if d["sites"]:
        st.markdown(f"### Citation sites of reference answer {sample}")
        rows = [
            {
                "sentence before the site": s["context"],
                "cited": s["cited"],
                f"P(target) {ref}": round(s["p_target_ref"], 3),
                f"P(target) {arm}": round(s["p_target_arm"], 3),
                "change": round(s["p_target_arm"] - s["p_target_ref"], 3),
            }
            for s in d["sites"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No AP rows for this reference answer (run `vizor score`, samples 0 and 1).")


def main() -> None:
    st.set_page_config(
        page_title="Vizor", page_icon=None, layout="wide", initial_sidebar_state="auto"
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="kicker">Vizor</div><div class="note">API <code>{html.escape(API)}</code></div>',
        unsafe_allow_html=True,
    )
    if os.environ.get("VIZOR_EGRESS_CANARY") == "1":
        state = egress_state()
        if state == "blocked":
            st.sidebar.markdown(f'<div class="note">egress: {state}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.warning(
                f"Offline mode requested, but outbound network is reachable ({state})."
            )
    pages = [
        st.Page(overview, title="Overview", default=True),
        st.Page(inspector, title="Answer inspector", url_path="inspector"),
        st.Page(side_by_side, title="Side by side", url_path="side-by-side"),
        st.Page(sandbox, title="Sandbox", url_path="sandbox"),
        st.Page(optimizer, title="Optimizer", url_path="optimizer"),
    ]
    st.navigation(pages).run()


main()
