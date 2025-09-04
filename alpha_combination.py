import numpy as np
import cvxpy as cp

def solve_two_trade_mpo_qp_dollar_neutral(w_t, r_fast, r_slow, Sigma,
                                          gamma_trade=1e-3, gamma_risk=1e-2,
                                          trade_bounds=None,
                                          solver=cp.ECOS):
    """
    Two-trade MPO with risk penalty, dollar-neutral and unit L1 norm constraints.
    
    Args:
      w_t : (N,) current weights
      r_fast, r_slow : (N,) alpha vectors
      Sigma : (N,N) covariance matrix
      gamma_trade : turnover penalty
      gamma_risk : risk penalty coefficient
      trade_bounds : (lb, ub) optional bounds for trades
    """
    N = len(w_t)
    z_short = cp.Variable(N)
    z_long  = cp.Variable(N)
    w_short = w_t + z_short
    w_final = w_short + z_long

    # Objective
    expected = r_fast @ w_short + r_slow @ w_final
    trade_cost = gamma_trade * (cp.norm1(z_short) + cp.norm1(z_long))
    risk_pen = 0.5 * gamma_risk * (cp.quad_form(w_short, Sigma) + cp.quad_form(w_final, Sigma))
    obj = cp.Maximize(expected - trade_cost - risk_pen)

    # Dollar-neutral, unit L1 norm constraints
    constraints = [
        cp.sum(w_short) == 0,
        cp.norm1(w_short) == 1,
        cp.sum(w_final) == 0,
        cp.norm1(w_final) == 1,
    ]

    if trade_bounds is not None:
        lb, ub = trade_bounds
        constraints += [z_short >= lb, z_short <= ub,
                        z_long  >= lb, z_long  <= ub]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)

    return {
        "status": prob.status,
        "objective": prob.value,
        "z_short": z_short.value,
        "z_long": z_long.value,
        "w_short": w_short.value,
        "w_final": w_final.value,
    }

# --- Example ---
np.random.seed(42)
N = 6
w_t = np.zeros(N)   # start neutral
r_fast = np.random.normal(0.001, 0.01, N)
r_slow = np.random.normal(0.002, 0.01, N)
Sigma = np.cov(np.random.randn(200, N).T)  # synthetic covariance

res = solve_two_trade_mpo_qp_dollar_neutral(w_t, r_fast, r_slow, Sigma,
                                            gamma_trade=5e-3, gamma_risk=0.1,
                                            trade_bounds=(-0.2, 0.2))

print("Status:", res["status"])
print("Objective:", res["objective"])
print("z_short:", np.round(res["z_short"], 4))
print("z_long:", np.round(res["z_long"], 4))
print("w_short:", np.round(res["w_short"], 4), "sum=", res["w_short"].sum(), "L1=", np.sum(np.abs(res["w_short"])))
print("w_final:", np.round(res["w_final"], 4), "sum=", res["w_final"].sum(), "L1=", np.sum(np.abs(res["w_final"])))
