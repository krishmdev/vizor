# Demo corpus

Everything in this folder is synthetic. The five sites are fictional brands on RFC 2606 `.example`
domains, written to give the sandbox a controlled corpus. Product names (Brewline Duo, Crema Lab
Aria, Kettleworks K3, and others), prices, and test numbers are invented.

| Domain | Role | Pages | Character |
|---|---|---|---|
| brewline.example | target | 5 | Manufacturer site with gaps on purpose: weak or missing meta descriptions, no FAQ, no JSON-LD, no internal links, key numbers buried at the bottom |
| crema-lab.example | competitor | 5 | Manufacturer with FAQ sections, FAQPage/Product JSON-LD, good descriptions and related links |
| shotreport.example | competitor | 5 | Review site heavy on measured numbers |
| beanbudget.example | competitor | 4 | Thin affiliate pages |
| grindandtamp.example | competitor | 3 | Hobbyist blog |

`queries.jsonl` contains 40 queries, 10 for each intent (informational, comparison,
transactional, troubleshooting). They were written by hand. `vizor queries generate` can build a
similar set from a real corpus with an LLM.
