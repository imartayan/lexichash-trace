import numpy as np


def _log4(n):
    return np.log(n) / np.log(4.0)


def _phi(k, n):
    """Exact O(k) recurrence for Phi_j, j = 1..k.

    Phi_j = c_j^2 + S_j / 3
    S_j   = (c_{j-1} - c_j)^2 + S_{j-1} / 4,   S_1 = 0
    """
    c = 1.0 - np.exp(-n * 4.0 ** (-np.arange(1, k + 2)))  # c_1 .. c_{k+1}
    Phi = np.empty(k)
    S = 0.0
    for j in range(1, k + 1):
        Phi[j - 1] = c[j - 1] ** 2 + S / 3.0
        if j < k:
            S = (c[j - 1] - c[j]) ** 2 + S / 4.0
    return Phi


def score(n, k, rho):
    """Exact mean LexicHash score (vectorized). rho: array of substitution rates."""
    rho = np.asarray(rho, dtype=float)
    j = np.arange(1, k + 1)[:, None]
    Phi = _phi(k, n)[:, None]
    w = 1.0 / (1.0 + (_log4(n) - 0.8) * rho)  # competition
    return (Phi + (1.0 - Phi) * w * np.exp(-j * rho)).sum(0)


def _moments(n, k):
    """O(k) precompute of the cumulant-matched model coefficients for (n, k).

    Approximates sum_j (1-Phi_j)*exp(-j*rho) by matching mean and variance
    of j under weights (1-Phi_j), rather than a Taylor expansion, which
    diverges once j*rho isn't small (e.g. already by rho=0.05 for k=31).
    """
    m = _log4(n) - 0.8
    Phi = _phi(k, n)
    e = 1.0 - Phi
    j = np.arange(1, k + 1, dtype=float)

    Sigma_Phi = Phi.sum()
    E0 = e.sum()
    mean_j = (j * e).sum() / E0
    var_j = ((j - mean_j) ** 2 * e).sum() / E0

    return m, Sigma_Phi, E0, mean_j, var_j


def score_approx(n, k, rho):
    """Approximate score(n, k, rho), accurate for rho in (0, ~0.1].

        score(rho) ~= Sigma_Phi + w(rho) * E0 * exp(-mean_j*rho + var_j*rho^2/2)

    Moment-matched single-exponential model (see _moments) combined with
    the exact competition factor w(rho) = 1/(1+m*rho). O(k) once per
    (n, k), then O(1) per rho. Relative error typically <0.1% for rho up
    to 0.1.
    """
    rho = np.asarray(rho, dtype=float)
    m, Sigma_Phi, E0, mean_j, var_j = _moments(n, k)

    w = 1.0 / (1.0 + m * rho)
    T = E0 * np.exp(-mean_j * rho + 0.5 * var_j * rho**2)
    return Sigma_Phi + w * T


def _score_and_deriv(n, k, rho):
    """`score(n, k, rho)` and its exact derivative d(score)/d(rho), both O(k)."""
    rho = np.asarray(rho, dtype=float)
    j = np.arange(1, k + 1)[:, None]
    Phi = _phi(k, n)[:, None]
    m = _log4(n) - 0.8
    w = 1.0 / (1.0 + m * rho)
    e = (1.0 - Phi) * np.exp(-j * rho)
    s = (Phi + e * w).sum(0)
    ds = (e * (-m * w**2 - j * w)).sum(0)
    return s, ds


def inverse(n, k, target_score, rho_max=1.0, bisect_iters=40, newton_iters=4):
    """Exact inverse of `score`: recover rho from (n, k, score).

    `score` is strictly decreasing in rho (each term is a product of positive
    decreasing factors), so the root is unique: bisect on [0, rho_max] for a
    robust seed (no dependence on the small-rho model behind `inverse_approx`),
    then polish with a few Newton steps against score's own exact derivative,
    which reach float64 precision from a bisected seed.
    """
    target_score = np.asarray(target_score, dtype=float)
    lo_bound, hi_bound = score(n, k, rho_max).item(), score(n, k, 0.0).item()
    if np.any(target_score <= lo_bound) or np.any(target_score > hi_bound):
        raise ValueError(
            f"score out of range for rho_max={rho_max}: expected "
            f"{lo_bound:g} < score <= {hi_bound:g}."
        )

    rho_lo = np.zeros_like(target_score)
    rho_hi = np.full_like(target_score, rho_max)
    for _ in range(bisect_iters):
        mid = 0.5 * (rho_lo + rho_hi)
        too_high = score(n, k, mid) > target_score
        rho_lo = np.where(too_high, mid, rho_lo)
        rho_hi = np.where(too_high, rho_hi, mid)
    rho = 0.5 * (rho_lo + rho_hi)

    for _ in range(newton_iters):
        s, ds = _score_and_deriv(n, k, rho)
        rho = np.clip(rho - (s - target_score) / ds, 0.0, None)

    return rho


def _seed(score, m, Sigma_Phi, E0, B1, B2):
    """Closed-form quadratic-in-log seed (see inverse_approx docstring)."""
    R = (score - Sigma_Phi) / E0
    if np.any(R <= 0) or np.any(R > 1 + 1e-9):
        raise ValueError(
            "score out of range for this model: expected "
            "Sigma_Phi < score <= k for rho in (0, ~0.1]."
        )
    R = np.minimum(R, 1.0)
    logR = np.log(R)

    disc = B1**2 + 2.0 * B2 * logR
    if np.any(disc < 0):
        raise ValueError(
            "score is out of range for the small-rho model (discriminant < 0)."
        )

    return (B1 - np.sqrt(disc)) / B2


def inverse_approx(n, k, score, newton_iters=2):
    """Invert score_approx: recover rho from (n, k, score), for rho in (0, ~0.1].

    Seed: linearize log(1+m*rho) to 2nd order in the score_approx exponent,
    making it quadratic in rho, and solve for R = (score-Sigma_Phi)/E0:

        rho_0 = (B1 - sqrt(B1^2 + 2*B2*log(R))) / B2,  B1=mean_j+m, B2=var_j+m^2

    That linearization caps the seed's own accuracy (~9% error near rho=0.1),
    so refine with `newton_iters` Newton steps against score_approx's exact
    closed form (cheap, O(1), since its derivative is analytic), bringing
    the error down to score_approx's own ~0.1-0.3% ceiling.
    """
    score = np.asarray(score, dtype=float)
    m, Sigma_Phi, E0, mean_j, var_j = _moments(n, k)
    B1, B2 = mean_j + m, var_j + m**2

    rho = _seed(score, m, Sigma_Phi, E0, B1, B2)

    for _ in range(newton_iters):
        w = 1.0 / (1.0 + m * rho)
        T = E0 * np.exp(-mean_j * rho + 0.5 * var_j * rho**2)
        f = Sigma_Phi + w * T - score
        fp = w * T * (var_j * rho - mean_j - m * w)
        rho = np.clip(rho - f / fp, 0.0, None)

    return rho


if __name__ == "__main__":
    n, k = 1_000_000, 31
    rho = np.array([0.0, 0.001, 0.01, 0.02, 0.05, 0.1])

    exact = score(n, k, rho)
    approx = score_approx(n, k, rho)
    print("rho          :", rho)
    print("score exact  :", np.round(exact, 6))
    print("score approx :", np.round(approx, 6))
    print("rel err %    :", np.round(100 * (exact - approx) / exact, 4))

    print()
    rho_from_exact = inverse_approx(n, k, exact)
    print("rho true     :", rho)
    print("rho recovered:", np.round(rho_from_exact, 6))
    print("abs err rho  :", np.round(rho - rho_from_exact, 6))
