# Response to MQL5 Practitioner Feedback

**Validated on 6,534 real GC=F 1-minute bars · May 11–15 2026 · Yahoo Finance**  
Full analysis: `docs/real_data_analysis.py` | `docs/real_data_analysis.md`

---

## Q1: Does the Hawkes kernel shape matter for exit decisions?

**No.** On real GC=F volume data (ACF = 0.358), the ratio heuristic with matched window W_s = 1/β achieves Spearman r = **0.475** against true Hawkes intensity — sufficient for threshold crossings. Across 53 real bursts, the matched window detected **15× faster** (0.94 min vs 14.1 min). Calibrate W_s = 1/β from your tick ACF.

---

## Q2: Is Kalman meaningfully better than EMA?

**Yes, for adverse-selection detection.** Tick SNR = **0.088** (spread-dominated). Across 203 real 30-minute trade windows, Kalman z-score FPR = **38%** vs EMA ATR FPR = **75%** — a **49% reduction**; **83%** in wide-spread periods. Replace ATR check with:

```
z = |p_filtered − p_entry| / √(P[0,0] + (spread/2)²)  >  2.0
```

---

## Q3: HJB boundary when adverse-selection and inventory gradients conflict near the spread

**Use the three-region policy.** Real GC=F data: q_min = **0.375** for a 2-tick target — positions below 37.5% of capacity should never exit. At Q = 0.40, **85% of ticks** are HJB-active. Do not exit if book imbalance favours you and |q| < q* = κ·imb/(2A).

```
|q| < q_min              →  never exit
q_min ≤ |q| < q*         →  exit only if Kalman z > 2
|q| ≥ q*                 →  exit when V(q,λ) + F > θ·exp(−t/τ)
```

---

## Supporting Evidence

| Claim | Prediction | Real GC=F result |
|---|---|---|
| Hawkes: matched window Spearman r > 0 | r > 0 | r = **0.475** ✓ |
| Hawkes: matched window faster detection | lower lag | **15× speedup** (0.94 vs 14.1 min) |
| Kalman: tick SNR spread-dominated | SNR ≪ 1 | SNR = **0.088** ✓ |
| Kalman: FPR reduction vs EMA | 40–55% | **49% overall, 83% wide-spread** |
| Spread: Gamma shape parameter | k ≈ 2.78 | k = **2.73** (2% error) |
| HJB: q_min linear in spread | Spearman 1.0 | r = **1.000** ✓ |
| HJB: no-exit zone active on real metals | significant | **dominant** for 2-tick GC target |
