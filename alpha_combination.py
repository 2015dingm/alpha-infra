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





# Option B: exact QP with pos/neg variables (4N), solved by scipy.optimize.minimize(trust-constr)
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, csc_matrix

def solve_exact_qp_posneg(w_t, r_fast, r_slow, Sigma,
                          gamma_trade=1e-3, gamma_risk=1e-2, opts=None):
    """
    Variables order: [z_plus(N), z_minus(N), wplus(N), wminus(N)] length = 4N
    z_short = z_plus - z_minus
    w_final = wplus - wminus
    Constraints (linear equalities):
      sum(w_short) == 0  -> sum(w_t) + sum(z_plus - z_minus) == 0
      sum(w_final) == 0  -> sum(wplus - wminus) == 0
      L1 constraints: sum(|w_short|) == 1 -> sum(z_plus + z_minus + abs_const)??? 
         But w_short = w_t + z_short; we need |w_short| = sum(|w_t + z_short|)
         This introduces absolute value of (w_t + z_short) -> not linear.
    ---
    To keep exactness for the required l1 norms of w_short and w_final we handle them as:
      enforce sum(|w_final|) = 1  --> sum(wplus + wminus) = 1  (since w_final = wplus - wminus)
      for w_short we cannot represent |w_short| exactly with only z_plus/z_minus unless w_t==0.
    ---
    Therefore this exact-QP approach assumes w_t == 0 (initial neutral). If w_t != 0, the L1 constraint on w_short becomes non-linear.
    For general w_t, Option B is best used when you only enforce the L1 constraint on w_final (common case).
    """
    N = len(w_t)
    # For simplicity we enforce dollar-neutral and unit-L1 on final only (common)
    # and enforce sum(w_short)=0 (linear) but not abs(w_short)==1 if w_t != 0.
    # If you need exact L1 on w_short with w_t != 0, we must add extra binaries (not done here).

    # variable indexing helpers
    def idx(n): return np.arange(n)
    z_p_idx = idx(N)
    z_m_idx = idx(N) + N
    w_p_idx = idx(N) + 2*N
    w_m_idx = idx(N) + 3*N
    nvar = 4*N

    # build quadratic objective 0.5 x^T H x + g^T x  (we will minimize)
    # Map variables to w_short and w_final expressions:
    # z_short = z_p - z_m
    # w_final = w_p - w_m
    # w_short = w_t + z_short

    # Construct linear term g from alphas and turnover
    # alpha: r_fast^T w_short + r_slow^T w_final
    # -> contributes linear coefficients: for z_p entries coefficient = -(r_fast) (since we minimize negative)
    # But careful: we will build g such that objective = 0.5 x^T H x + g^T x
    # Risk quadratic will build H; alphas and turnover (l1 via gamma * sum of pos/neg) are in g.

    # linear term from alpha:
    # d/d z_p of r_fast^T w_short = r_fast   (w_short depends on z_p positively)
    # same for z_m but with negative sign
    # for w_p/w_m, r_slow enters.
    g = np.zeros(nvar)
    # alpha contribution (we will minimize negative)
    # so add -r_fast for z_p and +r_fast for z_m
    g[z_p_idx] += -r_fast
    g[z_m_idx] += r_fast
    g[w_p_idx] += -r_slow
    g[w_m_idx] += r_slow

    # turnover L1 = sum(z_p + z_m) + sum(w_p + w_m)
    g += gamma_trade * np.concatenate([np.ones(2*N), np.ones(2*N)])

    # Quadratic risk: 0.5*gamma_risk*(w_short^T Sigma w_short + w_final^T Sigma w_final)
    # Expand to H: w_short = w_t + (z_p - z_m) => quadratic only in (z_p - z_m)
    # Let A_z be mapping from z variables to w_short: w_short = w_t + Mz where M maps z_p->+1, z_m->-1
    # Let B_w be mapping from wpos/wneg to w_final: w_final = M2 * w_vars where M2 maps w_p->+1, w_m->-1
    # Then risk quadratic matrix H = gamma_risk * (M^T Sigma M + M2^T Sigma M2)
    M = np.zeros((N, nvar))
    # z_p -> +1 at cols z_p_idx, z_m -> -1, w_p/w_m don't enter w_short
    M[:, z_p_idx] = np.eye(N)
    M[:, z_m_idx] = -np.eye(N)
    # w_final mapping
    M2 = np.zeros((N, nvar))
    M2[:, w_p_idx] = np.eye(N)
    M2[:, w_m_idx] = -np.eye(N)

    H = gamma_risk * (M.T @ Sigma @ M + M2.T @ Sigma @ M2)  # dense; for large N use sparse

    # define objective function (minimize)
    def fun(x):
        return 0.5 * x @ (H @ x) + g @ x

    # gradient
    def jac(x):
        return H @ x + g

    # constraints:
    # 1) sum(w_short) == 0  => sum(w_t) + sum(z_p - z_m) == 0 -> linear in x
    A1 = np.zeros(nvar)
    A1[z_p_idx] = 1.0
    A1[z_m_idx] = -1.0
    b1 = -np.sum(w_t)

    # 2) sum(w_final) == 0 -> sum(w_p - w_m) == 0
    A2 = np.zeros(nvar)
    A2[w_p_idx] = 1.0
    A2[w_m_idx] = -1.0
    b2 = 0.0

    # 3) unit L1 on final: sum(w_p + w_m) == 1
    A3 = np.zeros(nvar)
    A3[w_p_idx] = 1.0
    A3[w_m_idx] = 1.0
    b3 = 1.0

    # pack linear equality constraints
    Aeq = np.vstack([A1, A2, A3])
    beq = np.array([b1, b2, b3])

    # bounds: all variables >= 0 (pos/neg)
    bounds = [(0, None)] * nvar

    # use LinearConstraint
    from scipy.optimize import LinearConstraint
    lin_cons = LinearConstraint(Aeq, beq, beq)

    # Provide Hessian via callable (constant)
    def hess(x):
        return H

    res = minimize(fun, np.zeros(nvar), jac=jac, hess=hess,
                   constraints=[lin_cons], bounds=bounds, method='trust-constr', options=opts)

    if not res.success:
        raise RuntimeError("Solver failed: " + res.message)

    x = res.x
    z_plus = x[z_p_idx]; z_minus = x[z_m_idx]
    w_plus = x[w_p_idx]; w_minus = x[w_m_idx]

    z_short = z_plus - z_minus
    w_final = w_plus - w_minus
    w_short = w_t + z_short
    z_long = w_final - w_short

    # compute original (maximize) objective value
    alpha_term = r_fast.dot(w_short) + r_slow.dot(w_final)
    turnover = np.sum(np.abs(z_short)) + np.sum(np.abs(z_long))
    risk = 0.5 * gamma_risk * (w_short @ (Sigma @ w_short) + w_final @ (Sigma @ w_final))
    obj_val = alpha_term - gamma_trade * turnover - risk

    return {'z_short': z_short, 'z_long': z_long, 'w_short': w_short, 'w_final': w_final, 'obj': obj_val, 'success': res.success}
