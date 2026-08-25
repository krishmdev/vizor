# Study 2 primary (sampled citation share, page units) on 2026-09-18_study2-qwen3b

| Arm | dC-SoV pp [95% CI] | Holm p (Wilcoxon) | Holm p (sign-flip) | Pages | Zero pages | Verdict |
|---|---|---|---|---|---|---|
| `answer_first` | -9.5 [-15.8, -3.8] | 0.008 | 0.005 | 24 | 0 | effect |
| `evidence_surface_llm` | -4.3 [-9.1, -0.4] | 0.046 | 0.044 | 24 | 3 | effect |
| `faq_rewrite_v2` | -10.0 [-15.8, -4.8] | 0.004 | 0.003 | 24 | 4 | effect |
| `aa_resample` (control) | -3.4 [-6.6, -0.5] | 0.089 (raw) | 0.045 (raw) | 24 | 0 | control |
| `noop` (control) | +0.0 [+0.0, +0.0] | 1.000 (raw) | 1.000 (raw) | 24 | 24 | control |

MDE (A/A page SD 7.96 pp): 5.6 pp.
