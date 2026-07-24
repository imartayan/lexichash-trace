import numpy as np


def score(k, rho):
    """P(original best k-mer's hash is still the minimum) after mutation rate rho."""
    rho = np.asarray(rho, dtype=float)
    u = (1.0 - rho) ** k
    return u / (2 - u)


def fast_inverse(k, p, rho_max=1.0):
    """Invert score: recover rho from (k, p), exactly (score is invertible in closed form).

    p = u/(2-u)  =>  u = 2p/(1+p)  =>  rho = 1 - u^(1/k)

    p <= score(k, rho_max) is out of the invertible range (e.g. p=0 from an
    empirical match count of exactly 0, which the closed form would map to
    rho=1 with unwarranted confidence, since a floored sample is also
    consistent with any large-enough true rho): raise instead, same
    boundary as `inverse`.
    """
    p = np.asarray(p, dtype=float)
    lo_bound, hi_bound = score(k, rho_max), score(k, 0.0)
    if np.any(p <= lo_bound) or np.any(p > hi_bound):
        raise ValueError(
            f"score out of range for rho_max={rho_max}: expected "
            f"{lo_bound:g} < score <= {hi_bound:g}."
        )
    u = 2.0 * p / (1.0 + p)
    return 1.0 - u ** (1.0 / k)


def _score_and_deriv(k, rho):
    """`score(k, rho)` and its exact derivative d(score)/d(rho)."""
    rho = np.asarray(rho, dtype=float)
    u = (1.0 - rho) ** k
    s = u / (2.0 - u)
    du = -k * (1.0 - rho) ** (k - 1)
    ds = 2.0 * du / (2.0 - u) ** 2
    return s, ds


def inverse(k, p, rho_max=1.0, bisect_iters=60, newton_iters=4):
    """Invert score via bisection + Newton polish, same approach as
    `lexichash.inverse`, to check `fast_inverse`'s closed form for numerical
    imprecision: score is strictly decreasing in rho (u=(1-rho)^k is
    decreasing, and u/(2-u) is increasing in u), so the root is unique.
    """
    p = np.asarray(p, dtype=float)
    lo_bound, hi_bound = score(k, rho_max), score(k, 0.0)
    if np.any(p <= lo_bound) or np.any(p > hi_bound):
        raise ValueError(
            f"score out of range for rho_max={rho_max}: expected "
            f"{lo_bound:g} < score <= {hi_bound:g}."
        )

    rho_lo = np.zeros_like(p)
    rho_hi = np.full_like(p, rho_max)
    for _ in range(bisect_iters):
        mid = 0.5 * (rho_lo + rho_hi)
        too_high = score(k, mid) > p
        rho_lo = np.where(too_high, mid, rho_lo)
        rho_hi = np.where(too_high, rho_hi, mid)
    rho = 0.5 * (rho_lo + rho_hi)

    for _ in range(newton_iters):
        s, ds = _score_and_deriv(k, rho)
        rho = np.clip(rho - (s - p) / ds, 0.0, None)

    return rho


def old_score(k, rho):
    """Same as `score`, but with u = exp(-k*rho) (the small-k*rho limit of
    the exact (1-rho)^k) instead of the exact survival probability. Kept for
    comparison; biased by several percent once k*rho isn't small.
    """
    u = np.exp(-k * rho)
    return u / (2 - u)


def old_inverse(k, p):
    """Invert `old_score`: recover rho from (k, p) assuming u = exp(-k*rho).

    p = u/(2-u)  =>  u = 2p/(1+p)  =>  rho = -log(u) / k
    """
    p = np.asarray(p, dtype=float)
    u = 2.0 * p / (1.0 + p)
    with np.errstate(divide="ignore"):
        # p=0 (u=0) is a legitimate input (e.g. a small, all-miss sample)
        # and correctly yields rho=inf; suppress the resulting log(0) warning
        return -np.log(u) / k


if __name__ == "__main__":
    k = 31
    rho = np.array([0.0, 0.001, 0.01, 0.02, 0.05, 0.1])

    p = score(k, rho)
    p_old = old_score(k, rho)
    print("rho          :", rho)
    print("p (exact)    :", np.round(p, 6))
    print("p (old)      :", np.round(p_old, 6))
    print("rel err %    :", np.round(100 * (p_old - p) / p, 4))

    print()
    rho_recovered = inverse(k, p)
    print("rho true     :", rho)
    print("rho recovered:", np.round(rho_recovered, 6))
    print("abs err rho  :", np.round(rho - rho_recovered, 6))
