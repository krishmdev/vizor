# Study 2 primary (sampled citation share, page units) on 2026-09-18_study2-qwen3b

| Arm | dC-SoV pp [95% CI] | Holm p (Wilcoxon) | Holm p (sign-flip) | Pages | Zero pages | Verdict |
|---|---|---|---|---|---|---|
| `answer_first` | -9.5 [-15.8, -3.8] | 0.008 | 0.005 | 24 | 0 | effect |
| `evidence_surface_llm` | -4.3 [-9.1, -0.4] | 0.046 | 0.044 | 24 | 3 | effect |
| `faq_rewrite_v2` | -10.0 [-15.8, -4.8] | 0.004 | 0.003 | 24 | 4 | effect |
| `aa_resample` (control) | -3.4 [-6.6, -0.5] | 0.089 (raw) | 0.045 (raw) | 24 | 0 | control |
| `noop` (control) | +0.0 [+0.0, +0.0] | 1.000 (raw) | 1.000 (raw) | 24 | 24 | control |

MDE (A/A page SD 7.96 pp): 5.6 pp.

Exact sign-flip p (raw, all 2^24 sign patterns; the table uses the pre-registered Monte Carlo value with 20000 draws): `answer_first` 0.0028, `evidence_surface_llm` 0.0469, `faq_rewrite_v2` 0.0009, `aa_resample` 0.0456, `noop` 1.0000.

## Exploratory, not pre-registered: the choice of reference

Same page deltas and tests, Holm over the three arms, with the A/A re-sample or the mean of baseline and A/A as the reference instead of the baseline alone. This does not replace the verdicts above.

| Reference | Arm | dC-SoV pp | p Wilcoxon (Holm) | p sign-flip exact (Holm) | Both Holm p < 0.05 |
|---|---|---|---|---|---|
| A/A re-sample | `answer_first` | -6.1 | 0.046 (0.137) | 0.049 (0.140) | no |
| A/A re-sample | `evidence_surface_llm` | -0.9 | 0.812 (0.812) | 0.742 (0.742) | no |
| A/A re-sample | `faq_rewrite_v2` | -6.6 | 0.053 (0.137) | 0.047 (0.140) | no |
| mean of baseline and A/A | `answer_first` | -7.8 | 0.011 (0.029) | 0.009 (0.020) | yes |
| mean of baseline and A/A | `evidence_surface_llm` | -2.6 | 0.406 (0.406) | 0.287 (0.287) | no |
| mean of baseline and A/A | `faq_rewrite_v2` | -8.3 | 0.010 (0.029) | 0.007 (0.020) | yes |
