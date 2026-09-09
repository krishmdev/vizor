# Attribution propensity: 2026-09-18_study2-qwen3b

> **Gate failed; AP is secondary here and makes no claims.** Added after the run: this is the `vizor score` output, the same table as `ap/ap_summary.md`. AP failed its validation gate on criteria 3 and 4, so the "primary" family and "Significant" column below are not Study 2 verdicts; those are in `sampled_primary.md`.

Scorer `localhost-ai/qwen2.5-3b-mlx4`, pin `{"backend": "localhost-ai", "chat_template_kwargs": {}, "commit": "c61e05d4992d6cb68c7c16fc986eeb84dda74c5f", "model": "qwen2.5-3b-mlx4", "preset": "qwen2.5-3b-mlx4", "revision": "4f83f8f146fdf28b512a06562b671d7af4fab457", "server": "localhost-ai", "tokenizer_sha": "53532ad570e43718e4a465c9db068b743cf3bcae665b7608dbb80ed044b2b1d8", "url": "http://127.0.0.1:8431/v1/score"}`. Reference answers: baseline samples [0, 1]. 2016 scored (prompt set, query, reference) rows, 2003 with at least one citation site. Units are pages; the estimate, interval and tests use unweighted page means. AP deltas are in percentage points of citation probability.

| Comparison | Family | ΔAP pp [95% CI] | AP ref % | n queries / pages | p Wilcoxon (Holm) | p sign-flip (Holm) | Significant | ΔAP first site pp | Focus page shown % | Pages with Δ = 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pinned:answer_first` | primary | -1.34 [-3.00, +0.21] | 53.9 | 72 / 24 | 0.107 (0.215) | 0.118 (0.236) | no | -1.87 | 99 | 0 |
| `pinned:evidence_surface_llm` | primary | -0.04 [-0.75, +0.63] | 53.9 | 72 / 24 | 0.838 (0.838) | 0.905 (0.905) | no | +0.73 | 99 | 3 |
| `pinned:faq_rewrite_v2` | primary | -3.95 [-6.73, -1.62] | 53.9 | 72 / 24 | 0.00199 (0.00596) | 0.0011 (0.0033) | yes | -5.61 | 99 | 4 |
| `full:answer_first` | full | -1.61 [-5.03, +2.15] | 53.9 | 72 / 24 | 0.121 (0.242) | 0.411 (0.456) | no | -6.18 | 92 | 0 |
| `full:evidence_surface_llm` | full | -0.53 [-1.33, +0.25] | 53.9 | 72 / 24 | 0.432 (0.432) | 0.228 (0.456) | no | -3.35 | 97 | 3 |
| `full:faq_rewrite_v2` | full | -5.29 [-8.87, -2.44] | 53.9 | 72 / 24 | 0.000851 (0.00255) | 0.00045 (0.00135) | no | -9.14 | 99 | 4 |
| `rank:answer_first` | rank | -0.27 [-3.92, +3.80] | 52.6 | 72 / 24 | 0.404 (0.807) | 0.894 (0.987) | no | -4.31 | 92 | 8 |
| `rank:evidence_surface_llm` | rank | -0.48 [-1.28, +0.04] | 53.9 | 72 / 24 | 0.25 (0.75) | 0.25 (0.751) | no | -4.08 | 97 | 20 |
| `rank:faq_rewrite_v2` | rank | -1.34 [-4.02, +0.00] | 50.0 | 72 / 24 | 0.5 (0.807) | 0.494 (0.987) | no | -3.53 | 99 | 22 |
| `body-top3:answer_first` | body-top3 | -1.50 [-3.00, -0.21] | 55.1 | 72 / 24 | 0.0564 (0.169) | 0.0393 (0.118) | no | -1.67 | 99 | 0 |
| `body-top3:evidence_surface_llm` | body-top3 | -0.79 [-2.11, +0.30] | 55.1 | 72 / 24 | 0.0646 (0.169) | 0.254 (0.508) | no | +0.68 | 99 | 3 |
| `body-top3:faq_rewrite_v2` | body-top3 | +0.00 [+0.00, +0.00] | 55.1 | 72 / 24 | 1 (1) | 1 (1) | no | +0.00 | 99 | 24 |
| `aa_resample` | control | +0.00 [+0.00, +0.00] | 53.9 | 72 / 24 | 1 (1) | 1 (1) | no | +0.00 | 99 | 24 |
| `noop` | control | +0.00 [+0.00, +0.00] | 53.9 | 72 / 24 | 1 (1) | 1 (1) | no | +0.00 | 99 | 24 |
| `slot 5 vs 1` | slot | -22.39 [-30.74, -14.58] | 43.9 | 36 / 16 | 6.1e-05 (6.1e-05) | 0.0001 (0.0001) | no | -58.16 | 100 | 0 |

Significant: primary family only, both Holm-adjusted p values below 0.05. In full mode and the slot control the reference answer's indices are renumbered to each prompt's source order; the first-site column uses only each reference's first citation site, whose prefix holds no earlier marker.

