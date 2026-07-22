import numpy as np


def score(k, rho):
    """P(original best k-mer's hash is still the minimum) after mutation rate rho."""
    rho = np.asarray(rho, dtype=float)
    u = (1.0 - rho) ** k
    return u / (2 - u)


def inverse(k, p):
    """Invert score: recover rho from (k, p), exactly (score is invertible in closed form).

    p = u/(2-u)  =>  u = 2p/(1+p)  =>  rho = 1 - u^(1/k)
    """
    p = np.asarray(p, dtype=float)
    u = 2.0 * p / (1.0 + p)
    return 1.0 - u ** (1.0 / k)


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
