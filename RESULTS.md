# Industry ETF Systematic Strategy — Results

## Universe & Data

- **114 industry ETFs** (see `Industry_ETF_Tickers.csv`)
- **Price data**: `data/industry_etf_daily.parquet` — 2016-05-16 → 2025-12-31 (2,422 days)
- **IS window**: 2017-05-16 → 2025-12-31 (first valid signal after 252-day warmup)
- 10 tickers with <50% coverage (newer ETFs: BAI, CRPT, DRLL, FDND, FDRV, GDOC, GTEK, METV, NUKZ, SOXQ)

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

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | Calmar | SPY corr |
|---|---|---|---|---|---|---|
| ERC Momentum (L/S) | +15.20% | 23.63% | 0.643 | -41.59% | 0.365 | -0.145 |
| ERC Reversal (VIX>20) | +20.04% | 24.61% | 0.814 | -44.35% | 0.452 | +0.761 |
| **Combined Equal-Vol** | **+15.31%** | **14.09%** | **1.087** | **-21.44%** | **0.714** | **+0.459** |

Avg allocation: 47% Momentum / 53% Reversal.

**Why this works:**
- The VIX > 20 gate (active 34% of days) prevents the reversal from trading in low-vol regimes where mean-reversion is unreliable
- When reversal is flat (VIX ≤ 20), momentum carries the portfolio at low vol
- When reversal is active (VIX > 20), it provides large positive P&L that is partially uncorrelated with momentum
- The two strategies have daily P&L correlation of approximately **-0.15 to -0.25**, providing genuine diversification
- Combined vol (14.1%) is far below either component (23–25%), confirming near-additive diversification benefit

### Annual Returns — Combined (VIX>20 Gated)

| Year | Combined | Momentum | Reversal |
|---|---|---|---|
| 2018 | +2.5% | +2.2% | +1.3% |
| 2019 | +12.1% | +5.3% | +19.6% |
| 2020 | +49.1% | +65.2% | +55.0% |
| 2021 | +2.4% | -16.9% | +22.7% |
| 2022 | +36.2% | +55.3% | +10.8% |
| 2023 | +6.3% | -3.0% | +16.9% |
| 2024 | +13.6% | +22.3% | +7.4% |
| 2025 | +3.2% | -5.7% | +13.7% |

The complementary nature is clear: 2021 (momentum -16.9%, reversal +22.7%) and 2022 (momentum +55.3%, reversal +10.8%) are the clearest examples of the two legs offsetting each other.

---

## 4. Key Takeaways

1. **Momentum long leg is strong** (Sharpe 0.88); the short leg is a structural drag (-7.1% p.a.). The full L/S is still worthwhile for its hedging properties.

2. **Short-term reversal is a regime signal**, not an always-on strategy. It works when VIX > 20 (fear/dislocation) and fails in calm markets.

3. **Signal blending underperforms strategy blending** — combining at the signal level forces stocks to satisfy both criteria simultaneously, losing alpha from both edges. Running separate strategies and allocating at the portfolio level is superior.

4. **ERC is the best optimizer** for this universe — equal risk contribution handles the heterogeneous volatility of industry ETFs better than Min Var (too concentrated) or MDP (similar to Min Var in practice).

5. **Best standalone**: ERC Reversal VIX>20 (Sharpe 0.776–0.814 depending on basket size)  
   **Best combined**: ERC Momentum + ERC Reversal VIX>20 at equal-vol weights (Sharpe **1.087**, MaxDD **-21.4%**)

---

## 5. Output Files

| Directory | Contents |
|---|---|
| `results_momentum/` | Cumret plots, monthly heatmaps, stats CSV for all opt methods and quartile/tercile comparison |
| `results_reversal/` | AR(1) distribution, XS reversal cumrets, long-only reversal by N, VIX filter comparison, monthly heatmaps |
| `results_combined/` | Combined strategy cumret + allocation panel, monthly heatmaps, stats CSVs |

## 6. Scripts

| Script | Purpose |
|---|---|
| `download_history.py` | Download 10yr daily prices via yfinance → parquet |
| `momentum_backtest.py` | ERC/HRP/MinVar/MDP/EqualVol momentum backtest, quartile vs tercile |
| `reversal_analysis.py` | XS reversal, AR(1) analysis, long-only reversal, ERC reversal |
| `reversal_topn.py` | Long-only reversal for fixed N bottom names (1,3,5,10,15) |
| `reversal_topn_vix.py` | VIX-gated reversal across all N and thresholds (15, 20) |
| `reversal_quartile_vix20.py` | Bottom quartile/half reversal with VIX>20 gate |
| `signal_combined_erc.py` | Signal-level z-score blend → single ERC portfolio |
| `combined_strategy.py` | Strategy-level equal-vol combination (ungated reversal) |
| `combined_vix_gated.py` | Strategy-level equal-vol combination (VIX>20 gated reversal) ⭐ |
