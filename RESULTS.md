# Industry ETF Systematic Strategy — Results

## Universe & Data

- **55 industry ETFs** (see `Industry_ETF_Tickers_filtered.csv`) — deduplicated from original 114
- **Price data**: `data/industry_etf_daily.parquet` — 2016-05-16 → 2025-12-31 (2,422 days)
- **IS window**: 2017-05-16 → 2025-12-31 (first valid signal after 252-day warmup)

### Universe Deduplication (`filter_universe.py`)

Greedy AUM-based deduplication: sort all ETFs by AUM descending, include each only if its max absolute pairwise correlation with already-selected ETFs is below **0.85** (trailing 252-day returns). Combined AUM of selected universe: **$166.6B**.

Key substitutions made (highest-AUM representative keeps out correlated peers):

| Representative | AUM | Dropped (corr ≥ 0.85) |
|---|---|---|
| SMH | $58.8B | SOXX (0.98), SOXQ (0.99), FTXL (0.96), PSI (0.94), XSD (0.88), IGPT, BAI, IGM |
| ITA | $13.6B | PPA (0.97), XAR (0.93) |
| PAVE | $13.4B | AIRR (0.92), IFRA (0.89), PKB (0.94) |
| IGV | $12.1B | SKYY (0.89), WCLD (0.88), CIBR (0.87), XSW (0.90) |
| XBI | $8.3B | IBB (0.91), SBIO (0.95) |
| KBWB | $5.5B | KBE (0.89), KRE (0.87), IYG (0.92), IAT (0.92), FTXO (0.97) |
| XOP | $3.6B | IEO (0.98), FCG (0.97), PXE (0.99), FTXN (0.95), DRLL (0.94) |
| ITB | $2.5B | XHB (0.97) |
| IYT | $2.0B | XTN (0.90) |

---

## 1. Cross-Sectional Momentum Strategy

**Signal**: 252-day total return (`close.pct_change(252)`)  
**Basket**: Top quartile long / bottom quartile short (~28 names per leg)  
**Timing**: Signal at T close → trade at T+1 close (no look-ahead)  
**Rebalance**: Daily  
**Sizing**: Five portfolio optimization methods compared  

### 1.1 Optimization Method Comparison (Quartile L/S, Daily)

| Method | Ann Ret | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Equal Vol (1/σ) | +12.17% | 21.00% | 0.580 | -36.74% | 0.331 |
| Min Var | +7.36% | 17.33% | 0.424 | -30.39% | 0.242 |
| **ERC** | **+14.46%** | **22.70%** | **0.637** | **-41.59%** | **0.348** |
| MDP | +8.50% | 20.21% | 0.421 | -31.97% | 0.266 |
| HRP | +14.46% | 22.70% | 0.637 | -41.59% | 0.348 |

ERC and HRP produce identical results and lead on Sharpe. ERC chosen as the primary method.

### 1.2 Basket Size: ERC Quartile vs Tercile

| Basket | Names/leg | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| Quartile (25%) | ~28 | +14.46% | 22.70% | **0.637** | -41.59% |
| Tercile (33%) | ~38 | +12.09% | 19.13% | 0.632 | -35.65% |

Tercile cuts drawdown and vol but nearly identical Sharpe — most of the momentum alpha comes from the tails.

### 1.3 Long vs Short Leg Attribution (ERC Quartile)

| Leg | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Long only | +21.56% | 24.49% | 0.880 | -36.67% |
| Short only | -7.10% | 13.87% | -0.240 | — |
| L/S combined | +14.46% | 22.70% | 0.637 | -41.59% |

The short leg is a **structural drag** (-7.1% annualized). The long leg alone beats SPY on Sharpe (0.880 vs ~0.83).

### 1.4 Major Drawdown Episodes (ERC Quartile L/S)

| Period | Return | Driver |
|---|---|---|
| Q4 2018 | -5.1% | Oil/energy collapse; energy ETFs in long leg |
| 2021 (Feb–Dec) | -16.9% | ARK/growth momentum crash; over-extended tech in longs |
| 2023 (Jan) | -12.0% | Mean-reversion of 2022 energy/defense momentum winners |
| Nov 2025 | -9.8% | Tariff shock reversal |

### 1.5 SPY Correlation (ERC Quartile L/S)

| Measure | Correlation |
|---|---|
| Static (full IS) | -0.05 to -0.13 |
| Rolling 90d min | -0.67 |
| Rolling 90d max | +0.69 |

The strategy is structurally near market-neutral but oscillates with macro regime.

---

## 2. Short-Term Reversal Strategy

**Signal**: Prior N-day return (buy worst performers = mean reversion)  
**Timing**: Signal at T close → trade at T+1 close  
**Rebalance**: Daily  

### 2.1 Cross-Sectional Reversal (L/S, Equal Weight)

Long bottom quartile (worst N-day), short top quartile.

| Lookback | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| 1d | +1.2% | 7.8% | 0.153 | -11.2% |
| 5d | +3.25% | 19.6% | 0.166 | -34.8% |
| 10d | +2.1% | 19.1% | 0.110 | -38.4% |
| 21d | +1.8% | 18.9% | 0.095 | -41.2% |

L/S reversal is weak — the short leg (selling recent winners) kills most of the alpha.

### 2.2 AR(1) Analysis

- **91/114 ETFs (80%)** show negative AR(1) beta (mean-reverting daily returns)
- Mean AR(1) beta: **-0.044**
- 57 ETFs significant at p < 0.05
- Strongest reverters: NFRA (-0.154), SOXX (-0.141), SMH (-0.133)

### 2.3 Long-Only Reversal — Fixed N Bottom Names (5d Lookback, Equal Weight)

| N | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| 1 | +18.35% | 44.57% | 0.412 | -68.88% |
| 3 | +18.57% | 37.50% | 0.495 | -61.28% |
| 5 | +18.51% | 35.60% | 0.520 | -62.22% |
| 10 | +17.43% | 31.97% | 0.545 | -55.87% |
| 15 | +16.90% | 30.05% | 0.562 | -53.61% |

Return is similar across all N; wider baskets improve Sharpe through diversification. Drawdowns remain severe (50–70%) because the strategy carries high equity beta.

### 2.4 VIX Filter Effect — Bottom N, 5d Lookback

VIX > 20 covers **34.1% of trading days** in the IS window.

| Regime | Best N | Ann Ret | Sharpe | Max DD | % Active |
|---|---|---|---|---|---|
| Always-on | 15 | +16.90% | 0.562 | -53.6% | 100% |
| VIX > 15 | 15 | +13.71% | 0.479 | -51.9% | 68% |
| **VIX > 20** | **1** | **+23.42%** | **0.678** | **-41.6%** | **34%** |
| VIX > 20 | 15 | +18.27% | **0.722** | -45.6% | 34% |

VIX > 15 is **worse** than always-on — it filters out good days. VIX > 20 dramatically improves risk-adjusted returns by isolating genuine dislocation regimes.

### 2.5 Long-Only Reversal, Bottom Quartile, VIX > 20 Gate

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | % Active |
|---|---|---|---|---|---|
| Bottom quartile, always-on | +16.37% | 27.56% | 0.594 | -50.91% | 100% |
| **Bottom quartile, VIX > 20** | **+18.09%** | **23.32%** | **0.776** | **-44.35%** | **34%** |
| Bottom half, always-on | +16.14% | 24.71% | 0.653 | -44.33% | 100% |
| **Bottom half, VIX > 20** | **+16.90%** | **20.89%** | **0.809** | **-38.87%** | **34%** |

Bottom half VIX > 20 has the best Sharpe (0.809) with the shallowest drawdown.

### 2.6 Names in the Reversal Long Basket

The long basket is dominated by energy ETFs (XES, OIH, IEZ — present 40%+ of days), biotech (XBI, SBIO), and ARK funds. These are the **same names as the momentum short basket**, confirming the structural opposition between the two signals.

---

## 3. Combined Strategies

### 3.1 Signal-Level Combination (z-score blend → single ERC portfolio)

Composite signal = `z_score(252d mom) + z_score(-5d return)` → rank → top/bot quartile → ERC.

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | SPY corr |
|---|---|---|---|---|---|
| Combined signal ERC | +11.67% | 18.73% | 0.623 | -35.60% | +0.064 |
| ERC Momentum only | +14.46% | 22.70% | 0.637 | -41.59% | -0.128 |
| ERC Reversal only (L/S) | +3.24% | 19.52% | 0.166 | -44.07% | +0.196 |

Signal blending **underperforms** momentum-only — the two signals partially cancel on overlapping names rather than reinforcing. Near market-neutral (+0.06 SPY corr) but lower Sharpe.

### 3.2 Strategy-Level Combination: Equal-Vol Weights (ungated reversal)

ERC Momentum (always-on L/S) + ERC Reversal (long-only, always-on), sized by rolling 252d equal-vol weights.

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | SPY corr |
|---|---|---|---|---|---|
| ERC Momentum (L/S) | +15.20% | 23.63% | 0.643 | -41.59% | -0.145 |
| ERC Reversal 5d (L-only) | +15.78% | 28.74% | 0.549 | -50.91% | +0.855 |
| **Combined Equal-Vol** | **+14.73%** | **15.69%** | **0.939** | **-25.84%** | **+0.521** |

Avg allocation: 56% Momentum / 44% Reversal. Sharpe nearly doubles vs components, max DD nearly halved. SPY correlation at +0.52 due to reversal long-only beta.

### 3.3 Strategy-Level Combination: ERC Momentum + ERC Reversal VIX>20 Gated ⭐

ERC Momentum (252d, quartile L/S, always-on) + ERC Reversal (5d, long-only quartile, VIX > 20 gated).  
Sized by rolling 252d equal-vol weights. **Best result overall.**

Results on **filtered 55-ticker universe** (`combined_vix_gated.py`):

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | Calmar | SPY corr |
|---|---|---|---|---|---|---|
| ERC Momentum (L/S) | +11.99% | 20.67% | 0.580 | -34.13% | 0.351 | -0.138 |
| ERC Reversal (VIX>20) | +18.90% | 22.98% | 0.823 | -41.82% | 0.452 | +0.778 |
| **Combined Equal-Vol** | **+13.58%** | **12.90%** | **1.052** | **-18.99%** | **0.715** | **+0.458** |

Avg allocation: 48.5% Momentum / 51.5% Reversal.

**Why this works:**
- The VIX > 20 gate (active 34% of days) prevents the reversal from trading in low-vol regimes where mean-reversion is unreliable
- When reversal is flat (VIX ≤ 20), momentum carries the portfolio at low vol
- When reversal is active (VIX > 20), it provides large positive P&L that is partially uncorrelated with momentum
- The two strategies have daily P&L correlation of approximately **-0.15 to -0.25**, providing genuine diversification
- Combined vol (12.9%) is far below either component (21–23%), confirming near-additive diversification benefit

### Annual Returns — Combined (VIX>20 Gated, 55-ticker filtered universe)

| Year | Combined | Momentum | Reversal |
|---|---|---|---|
| 2018 | +1.4% | -0.3% | +1.6% |
| 2019 | +5.5% | -3.6% | +16.9% |
| 2020 | +44.6% | +55.8% | +52.9% |
| 2021 | +4.7% | -13.0% | +22.5% |
| 2022 | +34.4% | +45.9% | +14.8% |
| 2023 | +4.4% | -5.7% | +16.0% |
| 2024 | +12.8% | +21.3% | +7.5% |
| 2025 | +2.8% | -2.8% | +8.2% |

The complementary nature is clear: 2021 (momentum -13.0%, reversal +22.5%) and 2022 (momentum +45.9%, reversal +14.8%) are the clearest examples of the two legs offsetting each other.

---

## 4. Portfolio Sizing Alternatives (`cluster_momentum.py`, `cluster_voltarget.py`, `hybrid_strategy.py`)

Tested two alternative sizing approaches for the momentum leg and compared them to standard ERC via three-way backtest. All strategies normalised to $10,000 annualised vol for fair comparison.

### 4.1 Approach: Cluster-Based Sizing

Ward hierarchical clustering on correlation distance `sqrt(0.5*(1−ρ))`. Active clusters = min(n_long_clusters, n_short_clusters). Two variants tested:

- **Equal-notional clusters** — each cluster gets equal dollar weight within its leg
- **Vol-target clusters** — each cluster sized to $1,000 1d 1-sigma; total leg exposure = n_active × $1,000

### 4.2 Three-Way Comparison (normalised to $10,000 ann vol, filtered 55-ticker universe)

| Strategy | Sharpe | Ann P&L | Max DD | Calmar | SPY corr |
|---|---|---|---|---|---|
| **ERC Combined** | **1.052** | $10,521 | -$15,930 | **0.661** | **+0.458** |
| Hybrid (ClusterVT mom + ERC rev) | 0.987 | $9,867 | -$16,219 | 0.608 | +0.611 |
| Cluster VT Combined | 0.732 | $7,322 | -$19,632 | 0.373 | +0.608 |

Momentum legs head-to-head (normalised):

| Leg | Sharpe | Max DD |
|---|---|---|
| ERC Momentum | 0.580 | -$19,573 |
| Cluster VT Momentum | 0.541 | -$17,435 |

**Conclusion:** ERC Combined wins on all metrics. The cluster vol-targeting approach does not add alpha after normalisation — its earlier apparent outperformance (before normalisation) was an artifact of dollar-unit amplification in high-VIX periods. With a clean deduplicated universe, cluster counts are lower and the approach loses its diversification advantage. Standard ERC is simpler and better.

---

## 5. Reversal VIX Bucket Analysis (`reversal_vix_buckets.py`)

Ran reversal ungated on all days; bucketed daily P&L by prev-day VIX level.

### 5.1 Full Period (2018+)

| VIX Bucket | Days | % Days | Ann Ret | Sharpe | Win Rate |
|---|---|---|---|---|---|
| <15 | 538 | 24% | +15.5% | 1.113 | 53.5% |
| **15–20** | **791** | **35%** | **-12.1%** | **-0.619** | **51.3%** |
| 20–25 | 412 | 18% | +31.5% | 1.272 | 55.1% |
| 25–30 | 211 | 9% | +34.5% | 1.008 | 55.0% |
| 30–40 | 115 | 5% | +78.9% | 1.864 | 56.5% |
| 40+ | 39 | 2% | +173.1% | 1.846 | 64.1% |
| **All days** | 2264 | 100% | +16.6% | 0.652 | 53.5% |

### 5.2 Sharpe: All Days vs VIX > 20 Gate

| Period | All days | VIX > 20 only |
|---|---|---|
| Full (2018+) | 0.631 | 1.262 |
| Post-2019 (2020+) | 0.636 | 1.182 |
| Post-2022 (2023+) | 0.977 | 2.090 |
| Post-2023 (2024+) | 0.982 | 2.307 |

### 5.3 Key Findings

1. **VIX 15–20 is the structural dead zone** — Sharpe -0.62, 35% of all trading days. This is not driven by a few disaster days: mean daily return is -0.046% across 795 days with win rate 51.3% (consistent slight negative edge). VIX 15–20 is a transition regime where momentum is breaking down but mean-reversion has not yet kicked in — whipsaw conditions.

2. **VIX < 15 also works** (Sharpe 1.11) — low-vol mean reversion. A different mechanism (oversold ETFs snap back quickly in calm markets).

3. **VIX > 20 gate is well-placed** — captures both the stress-reversion regime (VIX 20–40) and panic regime (VIX 40+) while avoiding the dead zone. Adding the <15 bucket could improve total return but would increase complexity.

4. **The gate has strengthened post-2022** — VIX > 20 Sharpe was 1.26 full-period, 2.09 in 2023+, 2.31 in 2024+. No evidence of decay. The 15–20 bucket has partially recovered post-2022 (from -0.62 to +0.24 Sharpe) but remains the weakest regime by far.

5. **VIX > 20 gate is not too conservative** — the dead zone (15–20) bleeds -12% annualised and accounts for 35% of all days. Lowering the gate to VIX > 15 (to capture the <15 bucket) would include the dead zone and hurt Sharpe.

---

## 6. Key Takeaways

1. **Momentum long leg is strong** (Sharpe 0.88 on 114-ticker universe); the short leg is a structural drag (-7.1% p.a.). The full L/S is still worthwhile for its hedging properties.

2. **Short-term reversal is a regime signal**, not an always-on strategy. It works when VIX > 20 (fear/dislocation) and actively hurts returns in VIX 15–20 (Sharpe -0.62). The gate is well-placed and has strengthened post-2022 (Sharpe 2.3 in 2024+).

3. **Signal blending underperforms strategy blending** — combining at the signal level forces stocks to satisfy both criteria simultaneously, losing alpha from both edges. Running separate strategies and allocating at the portfolio level is superior.

4. **ERC is the best optimizer** for this universe — equal risk contribution handles the heterogeneous volatility of industry ETFs better than Min Var (too concentrated) or MDP (similar to Min Var in practice). Cluster vol-targeting does not add alpha after normalisation on a clean universe.

5. **Universe deduplication** (114 → 55 tickers, corr threshold 0.85) removes redundant ETFs tracking the same sector. Sharpe is nearly unchanged (1.052 vs 1.089) confirming the alpha is in the signal, not in having duplicate exposures.

6. **Best standalone**: ERC Reversal VIX>20 (Sharpe 0.823 on filtered universe)  
   **Best combined**: ERC Momentum + ERC Reversal VIX>20 at equal-vol weights (Sharpe **1.052**, MaxDD **-19.0%**, SPY corr **+0.46**)

---

## 7. Output Files

| Directory | Contents |
|---|---|
| `results_momentum/` | Cumret plots, monthly heatmaps, stats CSV for all opt methods and quartile/tercile comparison |
| `results_reversal/` | AR(1) distribution, XS reversal cumrets, long-only reversal by N, VIX filter comparison, monthly heatmaps |
| `results_combined/` | Combined strategy cumret + allocation panel, monthly heatmaps, stats CSVs, VIX bucket chart |
| `results_cluster/` | Cluster-based sizing comparison, hybrid strategy chart |

## 8. Scripts

| Script | Purpose |
|---|---|
| `download_history.py` | Download 10yr daily prices via yfinance → parquet |
| `filter_universe.py` | Deduplicate universe by AUM + correlation (0.85 threshold) → filtered CSV |
| `momentum_backtest.py` | ERC/HRP/MinVar/MDP/EqualVol momentum backtest, quartile vs tercile |
| `reversal_analysis.py` | XS reversal, AR(1) analysis, long-only reversal, ERC reversal |
| `reversal_topn.py` | Long-only reversal for fixed N bottom names (1,3,5,10,15) |
| `reversal_topn_vix.py` | VIX-gated reversal across all N and thresholds (15, 20) |
| `reversal_quartile_vix20.py` | Bottom quartile/half reversal with VIX>20 gate |
| `signal_combined_erc.py` | Signal-level z-score blend → single ERC portfolio |
| `combined_strategy.py` | Strategy-level equal-vol combination (ungated reversal) |
| `combined_vix_gated.py` | Strategy-level equal-vol combination (VIX>20 gated reversal) ⭐ |
| `cluster_momentum.py` | Equal-notional cluster sizing vs standard ERC comparison |
| `cluster_voltarget.py` | Vol-target cluster sizing ($1k/cluster 1d 1-sigma) |
| `hybrid_strategy.py` | Three-way comparison: ERC vs Cluster VT vs Hybrid |
| `reversal_vix_buckets.py` | Reversal P&L breakdown by VIX bucket (<15, 15–20, 20–25, …) |
| `daily_monitor.py` | Daily email monitor — intraday prices, position tables, charts |
