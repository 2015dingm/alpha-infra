"""
Alpha Splitter: decompose an existing strategy into
  (A) a low-turnover strategy that preserves most Sharpe
  (B) a high-turnover residual

Features
--------
1) Turnover-penalized projection of weights using convex optimization (L1 turnover) if cvxpy is available.
   - Tracks original weights while penalizing trading vs. last period.
   - Supports neutrality (sum w = 0) and box constraints (|w_i| <= cap).
   - Grid-search over lambda to target turnover or optimize net Sharpe under a cost model.
   - Always falls back to a fast closed-form L2 smoother if cvxpy is not available.

2) Signal filtering split (EWMA/Kalman-like): slow = EWMA(signal), fast = residual.
   - Works at the signal level; map to weights via your preferred mapping (e.g., w ∝ Σ^{-1}s).

3) Frequency-domain decomposition (Fourier-based):
   - Decompose signals into low-frequency (slow/low-turnover) and high-frequency (fast/high-turnover) components.
   - Useful for isolating long-horizon vs. short-horizon contributions.

4) Diagnostics & utilities
   - Turnover calculation, Sharpe (annualized), net returns with linear costs.
   - Optional orthogonalization (Gram–Schmidt) so the two legs are return-orthogonal.

Usage (minimal)
---------------
from alpha_splitter import (
    decompose_by_turnover_projection,
    decompose_by_signal_filtering,
    decompose_by_frequency,
    orthogonalize_returns,
    sharpe_annualized,
)

# Given arrays/dataframes
# R: (T x N) asset returns
# W_orig: (T x N) realized or target weights from your original strategy
# cost_bps: e.g., 5  (per unit turnover, roundtrip)

res = decompose_by_turnover_projection(R, W_orig,
                                       lambdas=[0.1, 0.3, 1.0, 3.0, 10.0],
                                       target_turnover=None,
                                       neutrality=True,
                                       box_cap=0.02,
                                       cost_bps=5,
                                       orthogonalize=True,
                                       ann_factor=252)

print(res.keys())  # ['W_LT','W_HT','ret_LT','ret_HT','ret_orig','grid','best_idx']
print(sharpe_annualized(res['ret_LT']))

Notes
-----
- R and weights should align in time; weights at t produce returns at t+1 typically.
- Turnover is computed as L1 change in weights per period (sum |Δw|).
- Costs are applied as: net_ret_t = gross_ret_t - (turnover_t * cost_per_unit).
- For production, replace the box_cap / neutrality with your house constraints and risk model.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

# ---- Optional dependency detection (cvxpy for L1 turnover with constraints) ----
try:
    import cvxpy as cp
    _HAS_CVX = True
except Exception:
    _HAS_CVX = False

# -------------------------- Basic utilities --------------------------

def _shift(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x
    out = np.empty_like(x)
    if k > 0:
        out[:k] = np.nan
        out[k:] = x[:-k]
    else:
        out[k:] = np.nan
        out[:k] = x[-k:]
    return out


def sharpe_annualized(ret: np.ndarray, ann_factor: int = 252, eps: float = 1e-12) -> float:
    ret = np.asarray(ret)
    mu = np.nanmean(ret)
    sd = np.nanstd(ret, ddof=1)
    if sd < eps:
        return 0.0
    return np.sqrt(ann_factor) * mu / sd


def turnover_L1(weights: np.ndarray) -> np.ndarray:
    W = np.asarray(weights)
    dW = W - _shift(W, 1)
    to = np.nansum(np.abs(dW), axis=1)
    to[0] = np.nan
    return to


def portfolio_returns(R: np.ndarray, W: np.ndarray, shift_weights: bool = True) -> np.ndarray:
    R = np.asarray(R)
    W = np.asarray(W)
    if shift_weights:
        W_eff = _shift(W, 1)
    else:
        W_eff = W
    port = np.nansum(W_eff * R, axis=1)
    return port


def apply_linear_costs(gross_ret: np.ndarray, turnover: np.ndarray, cost_bps: float) -> np.ndarray:
    c = cost_bps * 1e-4
    net = gross_ret - turnover * c
    return net


# ---------------- L2 fallback (closed-form partial adjustment) ----------------

def _l2_smoother_step(w_orig: np.ndarray, w_prev: np.ndarray, lam: float) -> np.ndarray:
    return (w_orig + lam * w_prev) / (1.0 + lam)


# ---------------- L1 (preferred) with optional constraints via cvxpy ----------------

def _l1_projection_step(
    w_orig: np.ndarray,
    w_prev: np.ndarray,
    lam: float,
    neutrality: bool = False,
    box_cap: Optional[float] = None,
) -> np.ndarray:
    if not _HAS_CVX:
        return _l2_smoother_step(w_orig, w_prev, lam)

    n = w_orig.shape[0]
    w = cp.Variable(n)
    obj = cp.sum_squares(w - w_orig) + lam * cp.norm1(w - w_prev)
    cons = []
    if neutrality:
        cons.append(cp.sum(w) == 0)
    if box_cap is not None:
        cons += [w <= box_cap, w >= -box_cap]
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w_orig) + lam * cp.norm1(w - w_prev)))
            prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return _l2_smoother_step(w_orig, w_prev, lam)

    if w.value is None:
        return _l2_smoother_step(w_orig, w_prev, lam)
    return np.asarray(w.value).reshape(-1)


# ---------------- Main: Turnover-penalized decomposition ----------------

def decompose_by_turnover_projection(
    R: np.ndarray,
    W_orig: np.ndarray,
    lambdas: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0),
    target_turnover: Optional[float] = None,
    neutrality: bool = False,
    box_cap: Optional[float] = None,
    cost_bps: float = 0.0,
    orthogonalize: bool = True,
    ann_factor: int = 252,
) -> Dict[str, Any]:
    R = np.asarray(R)
    W0 = np.asarray(W_orig)
    T, N = W0.shape

    results = []
    all_W_LT = []

    for lam in lambdas:
        W_LT = np.zeros_like(W0)
        W_prev = np.zeros(N)
        if neutrality:
            W_prev[:] = 0.0

        for t in range(T):
            w_targ = W0[t]
            W_prev = _l1_projection_step(w_targ, W_prev, lam, neutrality=neutrality, box_cap=box_cap)
            W_LT[t] = W_prev

        W_HT = W0 - W_LT

        ret_orig = portfolio_returns(R, W0)
        ret_LT_gross = portfolio_returns(R, W_LT)
        ret_HT_gross = portfolio_returns(R, W_HT)

        to_LT = turnover_L1(W_LT)
        to_HT = turnover_L1(W_HT)
        to_OR = turnover_L1(W0)

        ret_LT = apply_linear_costs(ret_LT_gross, to_LT, cost_bps)
        ret_HT = apply_linear_costs(ret_HT_gross, to_HT, cost_bps)
        ret_OR = apply_linear_costs(ret_orig, to_OR, cost_bps)

        if orthogonalize:
            x = ret_LT
            y = ret_HT
            x_c = x - np.nanmean(x)
            y_c = y - np.nanmean(y)
            varx = np.nansum(x_c * x_c)
            beta = 0.0 if varx == 0 else np.nansum(x_c * y_c) / varx
            y_orth = y - beta * x
            ret_HT_use = y_orth
        else:
            ret_HT_use = ret_HT

        grid_row = dict(
            lam=lam,
            mean_TO_LT=float(np.nanmean(to_LT)),
            mean_TO_OR=float(np.nanmean(to_OR)),
            sharpe_LT=float(sharpe_annualized(ret_LT, ann_factor)),
            sharpe_OR=float(sharpe_annualized(ret_OR, ann_factor)),
        )
        results.append(grid_row)
        all_W_LT.append(W_LT)

    if target_turnover is not None:
        idx = int(np.argmin([abs(r['mean_TO_LT'] - target_turnover) for r in results]))
    else:
        idx = int(np.argmax([r['sharpe_LT'] for r in results]))

    best_lam = lambdas[idx]
    W_LT = all_W_LT[idx]
    W_HT = W0 - W_LT
    ret_orig = portfolio_returns(R, W0)
    ret_LT_gross = portfolio_returns(R, W_LT)
    ret_HT_gross = portfolio_returns(R, W_HT)

    to_LT = turnover_L1(W_LT)
    to_HT = turnover_L1(W_HT)
    to_OR = turnover_L1(W0)

    ret_LT = apply_linear_costs(ret_LT_gross, to_LT, cost_bps)
    ret_HT = apply_linear_costs(ret_HT_gross, to_HT, cost_bps)
    ret_OR = apply_linear_costs(ret_orig, to_OR, cost_bps)

    if orthogonalize:
        x = ret_LT
        y = ret_HT
        x_c = x - np.nanmean(x)
        y_c = y - np.nanmean(y)
        varx = np.nansum(x_c * x_c)
        beta = 0.0 if varx == 0 else np.nansum(x_c * y_c) / varx
        ret_HT = y - beta * x

    out = {
        'W_LT': W_LT,
        'W_HT': W_HT,
        'ret_LT': ret_LT,
        'ret_HT': ret_HT,
        'ret_orig': ret_OR,
        'grid': results,
        'best_idx': idx,
        'best_lambda': best_lam,
    }
    return out


# ---------------- Signal-filtering split (EWMA) ----------------

def ewma(x: np.ndarray, halflife: Optional[float] = None, alpha: Optional[float] = None) -> np.ndarray:
    if alpha is None:
        if halflife is None:
            raise ValueError("Provide halflife or alpha")
        alpha = 1 - 0.5 ** (1.0 / halflife)
    x = np.asarray(x, float)
    out = np.empty_like(x)
    out[:] = np.nan
    s = None
    for t in range(x.shape[0]):
        xt = x[t]
        if np.any(np.isnan(xt)):
            out[t] = np.nan
            continue
        if s is None or np.any(np.isnan(s)):
            s = xt
        else:
            s = alpha * xt + (1 - alpha) * s
        out[t] = s
    return out


def decompose_by_signal_filtering(
    signal: np.ndarray,
    halflife: float = 10.0,
) -> Dict[str, np.ndarray]:
    s = np.asarray(signal)
    slow = ewma(s, halflife=halflife)
    fast = s - slow
    return {'slow': slow, 'fast': fast}


# ---------------- Frequency-domain decomposition ----------------

def decompose_by_frequency(signal: np.ndarray, cutoff: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Decompose a 1D signal into low-frequency and high-frequency components
    using FFT with a cutoff.

    Parameters
    ----------
    signal : array-like (T,)
    cutoff : float in (0, 0.5), fraction of Nyquist frequency

    Returns
    -------
    dict with 'low','high'
    """
    x = np.asarray(signal, float)
    n = len(x)
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    mask_low = freqs <= cutoff
    X_low = Xf * mask_low
    X_high = Xf * (~mask_low)
    low = np.fft.irfft(X_low, n=n)
    high = np.fft.irfft(X_high, n=n)
    return {'low': low, 'high': high}


# ---------------- Orthogonalization helper ----------------

def orthogonalize_returns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    x_c = x - np.nanmean(x)
    y_c = y - np.nanmean(y)
    varx = np.nansum(x_c * x_c)
    beta = 0.0 if varx == 0 else np.nansum(x_c * y_c) / varx
    return y - beta * x


# ---------------- Example (synthetic) ----------------
if __name__ == "__main__":
    np.random.seed(0)
    T, N = 750, 50
    R = 0.0005 + 0.01 * np.random.randn(T, N)
    base = np.random.randn(N)
    W_orig = np.zeros((T, N))
    w = base / np.sum(np.abs(base)) * 0.8
    for t in range(T):
        w = 0.98 * w + 0.02 * (np.random.randn(N))
        W_orig[t] = w + 0.02 * np.random.randn(N)
        W_orig[t] = np.clip(W_orig[t], -0.05, 0.05)
        W_orig[t] -= np.mean(W_orig[t])

    res = decompose_by_turnover_projection(
        R, W_orig,
        lambdas=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        neutrality=True,
        box_cap=0.05,
        cost_bps=5,
        orthogonalize=True,
        ann_factor=252,
    )

    print("Best lambda:", res['best_lambda'])
    print("Sharpe (orig, LT, HT):",
          sharpe_annualized(res['ret_orig']),
          sharpe_annualized(res['ret_LT']),
          sharpe_annualized(res['ret_HT']))
    print("Mean turnover (orig, LT, HT):",
          float(np.nanmean(turnover_L1(W_orig))),
          float(np.nanmean(turnover_L1(res['W_LT']))),
          float(np.nanmean(turnover_L1(res['W_HT']))))

    # Frequency decomposition example
    sig = np.sin(np.linspace(0, 20, T)) + 0.5 * np.random.randn(T)
    freq_parts = decompose_by_frequency(sig, cutoff=0.05)
    print("Freq-decomp Sharpe (low, high):",
          sharpe_annualized(freq_parts['low']),
          sharpe_annualized(freq_parts['high']))
