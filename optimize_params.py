"""
Offline parameter optimisation for OptimizedAgent.

Runs scipy.optimize.differential_evolution to maximise average PnL over
multiple market seeds, then saves the best parameter vector to best_params.npy.

Usage (from inside `make shell` or the container terminal):
    cd /abides
    python student_work/optimize_params.py

Knobs at the top of the file:
    SEEDS        — list of RMSC04 seeds used to average out noise
    END_TIME     — shortened trading day for faster evaluation
    POPSIZE      — DE population size (total evals ≈ POPSIZE*7*MAXITER)
    MAXITER      — DE iteration budget
    WORKERS      — -1 = use all CPU cores via multiprocessing
"""

import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

# Ensure repo root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abides_core import abides as abides_run
from abides_markets.configs.rmsc04 import build_config
from abides_markets.utils import config_add_agents

from student_work.optimized_agent import OptimizedAgent, _PARAM_KEYS, _PARAMS_FILE

# ── Optimisation settings ─────────────────────────────────────────────────────
SEEDS    = [42, 123, 7, 999, 2024]   # average over this many market seeds
END_TIME = "13:00:00"                 # half-day for speed; full run is 17:30:00
TICKER   = "ABM"
STARTING_CASH = 10_000_000           # cents

POPSIZE  = 5     # DE population multiplier  (total initial pop = POPSIZE * n_params)
MAXITER  = 30    # DE iteration limit
WORKERS  = -1    # -1 = use all CPU cores via multiprocessing; 1 = single-threaded (safer in notebooks)

# Parameter bounds  [window, w1, w2, w3, threshold, order_size, max_position]
BOUNDS = [
    (5,   60),    # window
    (0,    3),    # w1 — mean-reversion weight
    (0,    3),    # w2 — momentum weight
    (0,    3),    # w3 — inventory weight
    (0.05, 2.0),  # threshold
    (10,  150),   # order_size
    (100, 600),   # max_position
]
# ─────────────────────────────────────────────────────────────────────────────


def _run_single(theta: np.ndarray, seed: int) -> float:
    """Run one RMSC04 simulation with given params; return PnL in cents."""
    p = dict(zip(_PARAM_KEYS, theta))

    cfg = build_config(
        seed=seed,
        end_time=END_TIME,
        num_momentum_agents=0,
        log_orders=False,
        exchange_log_orders=False,
        book_logging=False,
        stdout_log_level="WARNING",
        starting_cash=STARTING_CASH,
    )

    base_id = len(cfg["agents"])
    agent = OptimizedAgent(
        id=base_id,
        name="OptAgent",
        type="OptimizedAgent",
        symbol=TICKER,
        starting_cash=STARTING_CASH,
        random_state=np.random.RandomState(seed=seed + 777),
        params=p,
        log_orders=False,
    )
    cfg = config_add_agents(cfg, [agent])

    end_state = abides_run.run(cfg)

    # Find the agent in the post-simulation state
    for a in end_state["agents"]:
        if a.id == base_id:
            return a.mark_to_market(a.holdings) - a.starting_cash

    return 0.0


def objective(theta: np.ndarray) -> float:
    """
    Objective for differential_evolution: return negative average PnL.
    Evaluated over all SEEDS so the optimiser sees the expected PnL, not
    a lucky single run.
    """
    pnls = [_run_single(theta, s) for s in SEEDS]
    avg  = float(np.mean(pnls))
    return -avg   # minimise → maximise PnL


_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
_iteration_count  = 0
_best_xk          = None
_no_improve_count = 0

EARLY_STOP_PATIENCE = 7     # stop if best params unchanged for this many iterations
EARLY_STOP_TOL      = 1e-4  # "unchanged" threshold (L2 distance in normalised param space)


def _checkpoint_callback(xk: np.ndarray, convergence: float) -> bool:
    global _iteration_count, _best_xk, _no_improve_count
    _iteration_count += 1

    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    iter_path = os.path.join(_CHECKPOINT_DIR, f"params_iter_{_iteration_count:04d}.npy")
    np.save(iter_path, xk)
    np.save(os.path.join(_CHECKPOINT_DIR, "params_latest.npy"), xk)

    # Early stopping: did the best solution move meaningfully?
    if _best_xk is None or np.linalg.norm(xk - _best_xk) > EARLY_STOP_TOL:
        _best_xk = xk.copy()
        _no_improve_count = 0
    else:
        _no_improve_count += 1

    stop = _no_improve_count >= EARLY_STOP_PATIENCE
    tag  = " — STOPPING (no improvement)" if stop else f"  no improvement streak: {_no_improve_count}/{EARLY_STOP_PATIENCE}"
    print(f"  [iter {_iteration_count:3d}] convergence={convergence:.4f} — checkpoint saved{tag}")
    return stop


def main() -> None:
    print(f"Starting optimisation — {len(SEEDS)} seeds × DE(popsize={POPSIZE}, maxiter={MAXITER})")
    print(f"  Short trading day: {END_TIME}   workers={WORKERS}")
    print(f"  Estimated evaluations: ~{POPSIZE * len(BOUNDS) * MAXITER * len(SEEDS)}")
    print(f"  Checkpoints → {_CHECKPOINT_DIR}/")
    print()

    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=42,
        popsize=POPSIZE,
        maxiter=MAXITER,
        tol=1e-3,
        workers=WORKERS,
        disp=True,
        polish=True,   # final L-BFGS-B pass over the best solution
        callback=_checkpoint_callback,
    )

    elapsed = time.time() - t0
    best_pnl_dollars = -result.fun / 100

    print()
    print(f"Optimisation finished in {elapsed:.1f}s")
    print(f"Best average PnL : ${best_pnl_dollars:+.2f}")
    print(f"Best params:")
    for k, v in zip(_PARAM_KEYS, result.x):
        print(f"  {k:15s} = {v:.4f}")

    np.save(_PARAMS_FILE, result.x)
    print(f"\nSaved → {_PARAMS_FILE}")

    # Invalidate class-level cache so next OptimizedAgent picks up new file
    OptimizedAgent._cached_params = None


if __name__ == "__main__":
    main()
