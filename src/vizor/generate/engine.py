"""The sandbox answer engine: cascade -> source-tagged prompt -> LLM -> parsed, cited answer."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vizor.attribution.citations import parse_answer
from vizor.generate.llm import LLM
from vizor.generate.prompt import format_prompt, prompt_hash, render_source
from vizor.optimize.retrieval_policy import RELEVANCE, RetrievalPolicy
from vizor.retrieve.cascade import Cascade, Selection
from vizor.types import Answer, Query, SourceRef, stable_seed


@dataclass
class EngineResult:
    answer: Answer
    prompt: str
    selection: Selection


class Engine:
    def __init__(
        self,
        cascade: Cascade,
        llm: LLM,
        temperature: float = 0.7,
        max_tokens: int = 450,
        base_seed: int = 0,
        wordcount: str = "alnum",
        workers: int = 1,
        system_prompt: str = "",
    ) -> None:
        self.system_prompt = system_prompt
        self.cascade = cascade
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_seed = base_seed
        self.wordcount = wordcount
        self.workers = workers

    def with_cascade(self, cascade: Cascade) -> Engine:
        return Engine(
            cascade,
            self.llm,
            self.temperature,
            self.max_tokens,
            self.base_seed,
            self.wordcount,
            self.workers,
            self.system_prompt,
        )

    def seed_for(self, query: Query, sample: int, salt: str = "") -> int:
        # Common random numbers: the seed depends only on (query, sample), never on the arm.
        return stable_seed(self.base_seed, query.query_id, sample, salt) & 0x7FFFFFFF

    def build_prompt(
        self, query: Query, policy: RetrievalPolicy = RELEVANCE
    ) -> tuple[str, Selection]:
        sel = self.cascade.select(query.text, policy)
        qvec = self.cascade.query_vector(query.text)
        rendered = [render_source(i + 1, c, self.cascade, qvec) for i, c in enumerate(sel.sources)]
        return format_prompt(query.text, rendered), sel

    def answer(
        self, query: Query, sample: int, policy: RetrievalPolicy = RELEVANCE, salt: str = ""
    ) -> EngineResult:
        return self._answer_from(query, sample, self.build_prompt(query, policy), salt)

    def run(
        self,
        queries: Iterable[Query],
        samples: int,
        policy: RetrievalPolicy = RELEVANCE,
        salt: str = "",
    ) -> list[EngineResult]:
        queries = list(queries)
        # Retrieval and prompt building run serially (the embedding caches aren't thread-safe);
        # only the LLM calls fan out, and only for remote backends.
        prompts = {q.query_id: self.build_prompt(q, policy) for q in queries}
        jobs = [(q, k) for q in queries for k in range(samples)]

        def call(job: tuple[Query, int]) -> EngineResult:
            q, k = job
            return self._answer_from(q, k, prompts[q.query_id], salt)

        if self.workers <= 1:
            return [call(j) for j in jobs]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(call, jobs))

    def _answer_from(
        self, query: Query, sample: int, built: tuple[str, Selection], salt: str
    ) -> EngineResult:
        prompt, sel = built
        seed = self.seed_for(query, sample, salt)
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        c = self.llm.complete(
            messages,
            temperature=self.temperature,
            seed=seed,
            max_tokens=self.max_tokens,
        )
        parsed = parse_answer(c.text, len(sel.sources), self.wordcount)
        refs = [
            SourceRef(
                i + 1,
                s.doc.doc_id,
                s.doc.domain,
                s.doc.url,
                round(s.retrieval_score, 6),
                round(s.rerank_score, 6),
                round(s.final_score, 6),
            )
            for i, s in enumerate(sel.sources)
        ]
        usage = {**c.usage, "cached": c.cached}
        if sel.forced is not None:
            usage["forced"] = sel.forced
        ans = Answer(
            query.query_id,
            sample,
            seed,
            c.model,
            prompt_hash(prompt),
            refs,
            c.text,
            parsed.sentences,
            parsed.hallucinated,
            usage,
        )
        return EngineResult(ans, prompt, sel)
