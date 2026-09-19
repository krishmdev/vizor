# Study 3 primary (sampled citation share, page units) on 2026-09-28_study3-qwen9b

| Arm | dC-SoV pp [95% CI] | Holm p (Wilcoxon) | Holm p (sign-flip) | Pages | Zero pages | Verdict |
|---|---|---|---|---|---|---|
| `fact_passage` | -5.5 [-10.3, -0.6] | 0.081 | 0.117 | 24 | 0 | no effect shown |
| `entity_anchor` | -1.7 [-3.4, -0.2] | 0.259 | 0.117 | 24 | 12 | no effect shown |
| `retrieval_meta` | -0.7 [-3.3, +2.0] | 0.565 | 0.626 | 24 | 0 | no effect shown |
| `aa_resample` (control) | +0.1 [-2.9, +3.1] | 0.989 (raw) | 0.946 (raw) | 24 | 0 | control |
| `noop` (control) | +0.0 [+0.0, +0.0] | 1.000 (raw) | 1.000 (raw) | 24 | 24 | control |

MDE (A/A page SD 7.73 pp): 5.4 pp.

Delta against the A/A arm (the rule needs the same sign as against the baseline): `fact_passage` -5.6 pp (same), `entity_anchor` -1.8 pp (same), `retrieval_meta` -0.8 pp (same).

Exact sign-flip p (raw, all 2^24 sign patterns; the table uses the pre-registered Monte Carlo value with 20000 draws): `fact_passage` 0.0394, `entity_anchor` 0.0547, `retrieval_meta` 0.6247, `aa_resample` 0.9463, `noop` 1.0000.

## Exploratory, not pre-registered: the choice of reference

Same page deltas and tests, Holm over the three arms, with the A/A re-sample or the mean of baseline and A/A as the reference instead of the baseline alone. This does not replace the verdicts above.

| Reference | Arm | dC-SoV pp | p Wilcoxon (Holm) | p sign-flip exact (Holm) | Both Holm p < 0.05 |
|---|---|---|---|---|---|
| A/A re-sample | `fact_passage` | -5.6 | 0.018 (0.054) | 0.081 (0.242) | no |
| A/A re-sample | `entity_anchor` | -1.8 | 0.241 (0.481) | 0.307 (0.613) | no |
| A/A re-sample | `retrieval_meta` | -0.8 | 0.390 (0.481) | 0.625 (0.625) | no |
| mean of baseline and A/A | `fact_passage` | -5.5 | 0.012 (0.035) | 0.050 (0.151) | no |
| mean of baseline and A/A | `entity_anchor` | -1.7 | 0.121 (0.242) | 0.126 (0.253) | no |
| mean of baseline and A/A | `retrieval_meta` | -0.8 | 0.360 (0.360) | 0.573 (0.573) | no |
