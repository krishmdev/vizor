# Attribution propensity: 2026-09-05_bench-qwen3b

> **Gate input, not a result.** Added after the run: these are the AP scores the validation gate read (`gate.md`), and AP failed the gate on criteria 3 and 4. No claim rests on them.

Scorer `localhost-ai/qwen2.5-3b-mlx4`, pin `{"backend": "localhost-ai", "chat_template_kwargs": {}, "commit": "c61e05d4992d6cb68c7c16fc986eeb84dda74c5f", "model": "qwen2.5-3b-mlx4", "preset": "qwen2.5-3b-mlx4", "revision": "4f83f8f146fdf28b512a06562b671d7af4fab457", "server": "localhost-ai", "tokenizer_sha": "53532ad570e43718e4a465c9db068b743cf3bcae665b7608dbb80ed044b2b1d8", "url": "http://127.0.0.1:8431/v1/score"}`. Reference answers: baseline samples [0, 1]. 2160 scored (prompt set, query, reference) rows, 1706 with at least one citation site. Units are pages; the estimate, interval and tests use unweighted page means. AP deltas are in percentage points of citation probability.

| Comparison | Family | ΔAP pp [95% CI] | AP ref % | n queries / pages | p Wilcoxon (Holm) | p sign-flip (Holm) | Significant | ΔAP first site pp | Focus page shown % | Pages with Δ = 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pinned:metadata` | pinned | +1.31 [+0.12, +2.83] | 41.1 | 64 / 23 | 0.119 (0.475) | 0.0587 (0.235) | no | +0.30 | 93 | 0 |
| `pinned:faq_rewrite` | pinned | -8.07 [-11.08, -5.38] | 41.1 | 64 / 23 | 1.19e-06 (7.15e-06) | 5e-05 (0.0003) | no | -6.90 | 93 | 0 |
| `pinned:jsonld_insert` | pinned | -0.91 [-3.06, +1.12] | 41.1 | 64 / 23 | 0.3 (0.9) | 0.435 (1) | no | -1.08 | 93 | 0 |
| `pinned:internal_links` | pinned | -2.20 [-4.01, -0.54] | 41.1 | 64 / 23 | 0.0254 (0.127) | 0.027 (0.135) | no | -2.69 | 93 | 0 |
| `pinned:stats_surface` | pinned | -0.49 [-1.76, +0.61] | 41.1 | 64 / 23 | 0.731 (1) | 0.462 (1) | no | -0.87 | 93 | 0 |
| `pinned:keyword_stuffing` | pinned | +0.30 [-1.34, +2.07] | 41.1 | 64 / 23 | 0.754 (1) | 0.745 (1) | no | +0.95 | 93 | 0 |
| `full:metadata` | full | +3.03 [+0.94, +5.34] | 41.1 | 64 / 23 | 0.0196 (0.0978) | 0.0138 (0.0692) | no | +2.43 | 92 | 0 |
| `full:faq_rewrite` | full | -8.41 [-11.44, -5.65] | 41.1 | 64 / 23 | 1.19e-06 (7.15e-06) | 5e-05 (0.0003) | no | -9.04 | 93 | 0 |
| `full:jsonld_insert` | full | -0.46 [-2.77, +1.76] | 41.1 | 64 / 23 | 0.501 (1) | 0.715 (1) | no | -1.08 | 94 | 0 |
| `full:internal_links` | full | -2.20 [-4.01, -0.54] | 41.1 | 64 / 23 | 0.0254 (0.102) | 0.027 (0.108) | no | -2.69 | 93 | 0 |
| `full:stats_surface` | full | +2.99 [-3.18, +10.63] | 41.1 | 64 / 23 | 0.893 (1) | 0.523 (1) | no | +1.84 | 89 | 0 |
| `full:keyword_stuffing` | full | +0.78 [-1.01, +2.73] | 41.1 | 64 / 23 | 0.377 (1) | 0.434 (1) | no | +0.97 | 93 | 0 |
| `rank:faq_rewrite` | rank | -0.34 [-2.48, +2.10] | 33.0 | 64 / 23 | 0.297 (0.297) | 0.765 (0.765) | no | -2.14 | 93 | 16 |
| `aa_resample` | control | +0.00 [+0.00, +0.00] | 41.1 | 64 / 23 | 1 (1) | 1 (1) | no | +0.00 | 93 | 23 |
| `slot 5 vs 1` | slot | -44.36 [-58.47, -29.70] | 56.6 | 33 / 17 | 0.000214 (0.000214) | 0.0002 (0.0002) | no | -71.78 | 100 | 0 |

Significant: primary family only, both Holm-adjusted p values below 0.05. In full mode and the slot control the reference answer's indices are renumbered to each prompt's source order; the first-site column uses only each reference's first citation site, whose prefix holds no earlier marker.
