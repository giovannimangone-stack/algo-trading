# Optimized Adaptive Trading Agent — ABIDES

A composite-signal trading agent for the [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) market simulator, developed for the *Algorithmic Trading* course (POLIMI Graduate School of Management, Master in Quantitative Finance) as part of the "Build and Battle Your Trading Agent" hackathon.

## Overview

`OptimizedAgent` is a `TradingAgent` subclass that wakes up every 60 seconds, reads the current bid/ask, and combines three normalized signals into a single trading decision:

- **Mean-Reversion Signal (MRS):** deviation of the mid-price from its rolling mean, in standard deviation units.
- **Momentum Signal (MS):** gap between a fast EMA (span 5) and a slow EMA (span 20).
- **Inventory Signal (IS):** penalizes directional exposure to keep the position balanced.

```
composite = w1 · MRS + w2 · MS + w3 · IS
```

If `composite > threshold`, the agent buys at the ask. If `composite < -threshold`, it sells at the bid. Otherwise it holds. Order sizes are capped by `max_position`.

A minimal baseline, `BasicAgent`, is included for comparison: it fires a single market buy at noon and stays idle for the rest of the day.

## Parameter Optimization

The seven parameters (`window, w1, w2, w3, threshold, order_size, max_position`) were tuned offline with `scipy.optimize.differential_evolution`, using the RMSC04 market configuration:

- **Fitness:** average mark-to-market PnL across 5 independent seeds (`42, 123, 7, 999, 2024`), evaluated over a half trading day (09:30–13:00).
- **Search space:** `window [5,60]`, `w1/w2/w3 [0,3]`, `threshold [0.05, 2.0]`, `order_size [10,150]`, `max_position [100,600]`.
- **DE setup:** population multiplier 5, max 30 iterations, early stopping after 7 generations without improvement, final L-BFGS-B polish.

The optimizer converged to:

| Parameter | Value | Interpretation |
|---|---|---|
| window | 30 | Rolling mean/std window (ticks) |
| w1 | 2.5 | Mean-reversion weight |
| w2 | 0 | Momentum weight (switched off) |
| w3 | 1.0 | Inventory penalty weight |
| threshold | 1.5 | Decision threshold |
| order_size | 30 | Shares per order |
| max_position | 550 | Maximum net inventory |

These values are the ones reported in the project's official course submission, and are hard-coded as defaults in `optimized_agent.py`. The agent will also load a `best_params.npy` file if one is present, so it can be re-optimized without touching the class.

## Limitations

Optimization ran on a **half trading day** with a **limited computational budget** (population multiplier 5, 30 iterations, 5 seeds). This was a deliberate speed/accuracy trade-off given the time available during the hackathon, not a claim of a fully validated strategy. Concretely:

- Parameters are fit to the specific seeds and time window used in training; behavior in the second half of the trading session (different liquidity and volatility patterns) was not directly optimized for.
- Results should be read as a proof of concept for the pipeline (signal design → DE tuning → deployment), not as a production-ready or generalizable trading strategy.
- `notebooks/test_full_day.ipynb` runs the agent on 30 full-day, out-of-training seeds, precisely to show honestly how it behaves outside the window it was tuned on.

## Repository Structure

```
.
├── README.md
├── basic_agent.py          # minimal baseline agent
├── optimized_agent.py       # OptimizedAgent (composite signal, loads best_params.npy if present)
├── optimize_params.py       # differential evolution tuning script
└── notebooks/
    ├── agent_monitor.ipynb   # single-seed trade visualization (price, inventory, intraday P&L)
    └── test_full_day.ipynb   # multi-seed, full-day out-of-sample test
```

## Framework Notes

Built on [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) (Byrd et al., 2020), a discrete-event market simulator. The kernel dispatches timestamped messages between agents (an `ExchangeAgent` running the limit order book, `NoiseAgent`, `ValueAgent`, `MomentumAgent`, `MarketMakerAgent`, and the student agent) without advancing time in fixed steps. The RMSC04 configuration reproduces a realistic single-ticker equity trading day, with a mean-reverting fundamental value and a background population calibrated to produce realistic spread, depth, and volume.

Optimization method: Differential Evolution (Storn & Price, 1997), a population-based, gradient-free global optimizer well suited to the noisy, non-differentiable objective of a simulator-based fitness function.

## References

- Byrd, D., Hybinette, M., & Balch, T. (2020). *ABIDES: Towards high-fidelity market simulation for AI research.* ACM ICAIF.
- Storn, R., & Price, K. (1997). *Differential Evolution — A simple and efficient heuristic for global optimization over continuous spaces.* Journal of Global Optimization, 11(4), 341–359.
