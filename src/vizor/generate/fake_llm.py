"""FakeLLM: a deterministic, offline "extractive citing" stand-in for a real model.

It exists so the whole pipeline (retrieval, prompt, citation parsing, metrics, sandbox, bandit)
runs in CI and in the keyless demo. Its numbers are a pipeline check, not evidence about how any
real model behaves: it picks source sentences by embedding similarity to the question, so it
rewards lexical overlap and keyword stuffing "works" on it.

For each candidate sentence s (6 to 40 words) in source i:
    score = cos(q, s) - gamma * (i - 1) + tau * Gumbel
gamma = 0.05 is a mild position prior (primacy bias, as in Liu et al. 2023, "Lost in the Middle")
and tau = 0.1 is the sampling temperature. The top m in {4..6} sentences are kept (near-duplicates
with cos > 0.9 dropped), sentences from different sources with cos > 0.75 are merged into one
sentence citing both, and the answer is written in score order.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from vizor.attribution.citations import split_sentences
from vizor.embed import Embedder
from vizor.generate.llm import Completion, Messages
from vizor.generate.prompt import parse_prompt

GAMMA = 0.05
TAU = 0.1
DUP_COS = 0.9
MERGE_COS = 0.75


def _candidates(content: str, description: str) -> list[str]:
    out = []
    for line in [description, *content.split("\n")]:
        line = line.strip()
        if not line or line.startswith("Related:"):
            continue
        if line.startswith("Q: ") and " A: " in line:
            line = line.split(" A: ", 1)[1]
        for s in split_sentences(line):
            s = s.rstrip(" …")
            if 6 <= len(s.split()) <= 40:
                out.append(s)
    return out


class FakeLLM:
    def __init__(self, embedder: Embedder, gamma: float = GAMMA, tau: float = TAU) -> None:
        self.embedder = embedder
        self.gamma = gamma
        self.tau = tau
        self.model_id = f"fake-extractive/v1/{embedder.embedder_id.split('/')[1]}"

    def complete(
        self, messages: Messages, *, temperature: float, seed: int, max_tokens: int
    ) -> Completion:
        prompt = messages[-1]["content"]
        question, sources = parse_prompt(prompt)
        qh = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], "little")
        rng = np.random.Generator(np.random.PCG64(seed ^ qh))

        cands: list[tuple[int, str]] = []
        for src in sources:
            cands += [(src.index, s) for s in _candidates(src.content, src.description)]
        if not cands:
            return Completion("The provided sources do not answer this question.", self.model_id)

        qv = self.embedder.encode([question], kind="query")[0]
        sv = self.embedder.encode([s for _, s in cands], kind="passage")
        prior = np.array([self.gamma * (i - 1) for i, _ in cands])
        score = sv @ qv - prior + self.tau * rng.gumbel(size=len(cands))
        m = int(rng.integers(4, 7))

        chosen: list[int] = []
        for j in np.argsort(-score, kind="stable"):
            if any(float(sv[j] @ sv[k]) > DUP_COS for k in chosen):
                continue
            chosen.append(int(j))
            if len(chosen) == m:
                break

        merged: list[tuple[str, list[int]]] = []
        used: set[int] = set()
        for a in chosen:
            if a in used:
                continue
            cites = [cands[a][0]]
            for b in chosen:
                fresh = b != a and b not in used and cands[b][0] not in cites
                if fresh and float(sv[a] @ sv[b]) > MERGE_COS:
                    cites.append(cands[b][0])
                    used.add(b)
            used.add(a)
            merged.append((cands[a][1], cites))

        parts = []
        for text, cites in merged:
            text = re.sub(r"[.!?]+$", "", text.strip())
            parts.append(text + " " + "".join(f"[{c}]" for c in cites) + ".")
        n_words = sum(len(p.split()) for p in parts)
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": n_words}
        return Completion(" ".join(parts), self.model_id, usage)
