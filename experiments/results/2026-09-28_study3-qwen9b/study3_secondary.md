# Study 3 secondary metrics on 2026-09-28_study3-qwen9b (no claims; raw p values)

Baseline: focus page shown in 100% of queries, mean candidate rank 1.74.

| Arm | Named rate | Full dC-SoV | Content-only | Rank-mediated | Retrieved | Rank |
|---|---|---|---|---|---|---|
| `fact_passage` | -5.4 [-16.1, +3.0] pp (p 0.341) | -5.5 [-10.3, -0.6] pp (p 0.027) | -1.5 [-5.8, +2.9] pp (p 0.208) | -3.9 [-8.3, -0.1] pp (p 0.152) | -4.9 [-12.5, +0.0] pp (p 0.500) | +0.6 [+0.2, +1.1] ranks (p 0.003) |
| `entity_anchor` | +1.2 [-3.1, +5.6] pp (p 0.493) | -1.7 [-3.4, -0.2] pp (p 0.129) | +0.2 [-2.1, +2.2] pp (p 0.406) | -1.8 [-3.5, -0.1] pp (p 0.065) | -1.0 [-3.1, +0.0] pp (p 1.000) | +0.0 [+0.0, +0.0] ranks (p 1.000) |
| `retrieval_meta` | +0.2 [-5.9, +5.0] pp (p 0.671) | -0.7 [-3.3, +2.0] pp (p 0.565) | -0.7 [-3.9, +2.7] pp (p 0.584) | -0.0 [-2.0, +1.8] pp (p 0.899) | +0.0 [+0.0, +0.0] pp (p 1.000) | -0.1 [-0.3, +0.1] ranks (p 0.096) |

Passage selection: share of queries with an edited or new passage of the page among those shown (all queries; queries where the page is shown).

| Set | Page shown | Mean rank | Edited/new shown (all) | Edited/new shown (if shown) |
|---|---|---|---|---|
| `fact_passage` | 96% | 2.27 | 96% | 100% (69) |
| `content:fact_passage` | 100% | 1.74 | 100% | 100% (72) |
| `entity_anchor` | 99% | 1.72 | 44% | 45% (71) |
| `content:entity_anchor` | 100% | 1.74 | 46% | 46% (72) |
| `retrieval_meta` | 100% | 1.56 | 0% | 0% (72) |
| `content:retrieval_meta` | 100% | 1.74 | 0% | 0% (72) |
