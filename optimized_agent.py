
"""
Optimized Adaptive Trading Agent

Strategy: composite signal from three normalized features, whose weights and
thresholds are tuned offline by scipy.optimize.differential_evolution.

Signals:
  1. Mean-reversion (MRS): price below rolling mean → buy; above → sell
  2. Momentum      (MS):   fast EMA > slow EMA     → buy; below  → sell
  3. Inventory     (IS):   large long position      → lean sell; short → lean buy

Decision:
  composite = w1 * MRS + w2 * MS + w3 * IS
  composite >  threshold → BUY  order_size shares (limit @ ask for fast fill)
  composite < -threshold → SELL order_size shares (limit @ bid for fast fill)
  else                   → HOLD

Optimised parameters (loaded from best_params.npy if present):
  window, w1, w2, w3, threshold, order_size, max_position
"""

import os
from collections import deque
from typing import Dict, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


# Fixed hypers (not optimised — change here if needed)
_FAST_SPAN = 5
_SLOW_SPAN = 20
_WAKE_FREQ = str_to_ns("60s")

# Fall-back parameters used when no best_params.npy file exists
_DEFAULT_PARAMS: Dict[str, float] = {
    "window":       30,
    "w1":           2.5,   # mean-reversion weight
    "w2":           0,   # momentum weight
    "w3":           1,   # inventory weight
    "threshold":    1.5,
    "order_size":   30,
    "max_position": 550,
}

# Ordered list matching the numpy array saved by the optimiser
_PARAM_KEYS = ["window", "w1", "w2", "w3", "threshold", "order_size", "max_position"]

_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.npy")


def _load_params() -> Dict[str, float]:
    if os.path.exists(_PARAMS_FILE):
        raw = np.load(_PARAMS_FILE)
        return dict(zip(_PARAM_KEYS, raw.tolist()))
    return dict(_DEFAULT_PARAMS)


class OptimizedAgent(TradingAgent):
    """
    Adaptive multi-signal agent whose decision parameters are tuned by
    scipy.optimize (see optimize_params.py).  Deploy multiple instances
    for the tournament — all share the same loaded policy.
    """

    # Cache loaded params across instances so we only hit disk once
    _cached_params: Optional[Dict[str, float]] = None

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        params: Optional[Dict] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol

        # Resolve parameters: explicit arg > file > hard-coded defaults
        if params is not None:
            p = params
        else:
            if OptimizedAgent._cached_params is None:
                OptimizedAgent._cached_params = _load_params()
            p = OptimizedAgent._cached_params

        self.window: int       = max(5, int(round(p["window"])))
        self.w1: float         = float(p["w1"])
        self.w2: float         = float(p["w2"])
        self.w3: float         = float(p["w3"])
        self.threshold: float  = float(p["threshold"])
        self.order_size: int   = max(1, int(round(p["order_size"])))
        self.max_position: int = max(1, int(round(p["max_position"])))

        # Price history (bounded deque; keeps max of window or slow span)
        _buf = max(self.window, _SLOW_SPAN) + 10
        self._prices: deque = deque(maxlen=_buf)

        # EMA state (updated incrementally)
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._fast_alpha: float = 2.0 / (_FAST_SPAN + 1)
        self._slow_alpha: float = 2.0 / (_SLOW_SPAN + 1)

        self.state: str = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def get_wake_frequency(self) -> NanosecondTime:
        return _WAKE_FREQ

    def wakeup(self, current_time: NanosecondTime) -> None:
        can_trade = super().wakeup(current_time)
        if can_trade:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD" and isinstance(message, QuerySpreadResponseMsg):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                self._tick(bid, ask)
            self.set_wakeup(current_time + _WAKE_FREQ)
            self.state = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Signal computation and order placement
    # ------------------------------------------------------------------

    def _update_emas(self, mid: float) -> None:
        if self._fast_ema is None:
            self._fast_ema = mid
            self._slow_ema = mid
        else:
            self._fast_ema = self._fast_alpha * mid + (1 - self._fast_alpha) * self._fast_ema
            self._slow_ema = self._slow_alpha * mid + (1 - self._slow_alpha) * self._slow_ema

    def _tick(self, bid: int, ask: int) -> None:
        mid = (bid + ask) / 2.0
        self._prices.append(mid)
        self._update_emas(mid)

        if len(self._prices) < self.window:
            return

        recent = list(self._prices)[-self.window:]
        roll_mean = float(np.mean(recent))
        roll_std  = float(np.std(recent)) or 1.0

        # Signal 1: mean reversion — positive when price below mean (→ buy)
        mrs = (roll_mean - mid) / roll_std

        # Signal 2: momentum — positive in uptrend (fast > slow → buy)
        ms = (self._fast_ema - self._slow_ema) / (self._slow_ema / 1_000.0)

        # Signal 3: inventory — positive when short/flat (→ lean buy)
        pos = self.holdings.get(self.symbol, 0)
        inv_s = -pos / float(self.max_position)

        composite = self.w1 * mrs + self.w2 * ms + self.w3 * inv_s

        if composite > self.threshold and pos < self.max_position:
            qty = min(self.order_size, self.max_position - pos)
            if qty > 0:
                # Buy limit at ask price → crosses spread, fills immediately
                self.place_limit_order(
                    self.symbol, quantity=qty, side=Side.BID, limit_price=ask
                )

        elif composite < -self.threshold and pos > -self.max_position:
            qty = min(self.order_size, self.max_position + pos)
            if qty > 0:
                # Sell limit at bid price → crosses spread, fills immediately
                self.place_limit_order(
                    self.symbol, quantity=qty, side=Side.ASK, limit_price=bid
                )
