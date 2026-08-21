# Study 2 validation gate on 2026-09-05_bench-qwen3b: FAILED

| Criterion | Result | Measured | Threshold |
|---|---|---|---|
| 1. AP detects slot 5 vs 1 | pass | -44.36 pp, Holm p 0.0002 (sign-flip) / 0.00021 (Wilcoxon), 17 pages | negative, both Holm p < 0.01 |
| 2. FAQ rewrite negative | pass | -8.07 pp [-11.08, -5.38] | < 0 |
| 3. Relative precision, FAQ rewrite | FAIL | AP abs(d)/half-width 2.83 vs C-SoV 2.24 (ratio 0.79) | C-SoV / AP <= 0.6 |
| 4. Spearman, AP vs C-SoV page deltas | FAIL | rho 0.25 over 138 points, / sqrt(rel 0.284) = 0.47, one-sided page-permutation p 0.335 | corrected >= 0.5, rho > 0, p < 0.05 |

Study 2 MDE on AP (t quantiles, 80% power, two-sided alpha 0.05 / family, 24 pages, family 3): 2.09 to 5.07 pp, from page-level SDs of 2.97 to 7.22 pp.
