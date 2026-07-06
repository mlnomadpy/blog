"""The regularization price list, computed for real.

Companion experiment for "Why Regularization Is a Price List". There is NO
gradient-descent training loop anywhere in this file: kernel ridge on a small
dataset is an analytic solve, (K + lambda I) alpha = y, plus one symmetric
eigendecomposition of the Gram matrix. Everything the companion post and its
GIFs display is exported to public/regularization/ as JSON from this run.

What it computes, as a function of the ridge budget lambda:

  * the representer coefficients alpha = (K + lambda I)^{-1} y            (the solve)
  * the RKHS-norm bill  ||f||^2 = alpha^T K alpha                        (the budget spent)
  * the effective dimension  d_eff = sum_k lambda_k / (lambda_k + lambda) (modes that survive)
  * the eigenvalue spectrum lambda_k of the Gram matrix                   (the price list)
  * per-point |alpha_i|, the graded (NOT sparse) shrink                   (honest ridge)
  * a train/test generalization curve vs lambda                          (the point of the post)
  * the points<->modes bridge: c_k = lambda_k * sum_i alpha_i phi_k(x_i)  (the exchange rate)

Two datasets:
  A. a 1-D regression toy (a real sinc-like target sampled with noise) that carries
     the generalization U-curve, the spectrum, d_eff, the bill, and the c_k bridge.
  B. the six-point, two-class miniature from the explainer, for the |alpha_i| bars
     and the assembly order.

Run: python scripts/regularization_pricelist.py
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "regularization"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# The kernel. A Gaussian (RBF) kernel: a bump largest at coincidence, decaying
# with squared distance. It gives <phi(x), phi(y)> without ever forming phi.
# --------------------------------------------------------------------------- #
def rbf(A, B, sig):
    d2 = (A[:, None, :] - B[None, :, :]) ** 2
    return np.exp(-d2.sum(-1) / (2.0 * sig ** 2))


def kernel_ridge(K, y, lam):
    """The representer solve: (K + lambda I) alpha = y. No optimization loop."""
    n = K.shape[0]
    return np.linalg.solve(K + lam * np.eye(n), y)


def rkhs_norm_sq(K, alpha):
    """The bill for the solution: ||f||^2_H = alpha^T K alpha."""
    return float(alpha @ (K @ alpha))


def eff_dim(eigs, lam):
    """d_eff = sum_k lambda_k / (lambda_k + lambda): the modes that survive lambda."""
    return float(np.sum(eigs / (eigs + lam)))


# --------------------------------------------------------------------------- #
# Dataset A: a real 1-D regression toy. Target is a damped-sine (sinc-like)
# function; we sample it with noise and split into train/test.
# --------------------------------------------------------------------------- #
def build_regression():
    rng = np.random.default_rng(0)
    target = lambda x: np.sin(2.6 * x) * np.exp(-0.18 * x ** 2)

    n_tr, n_te = 40, 400
    x_tr = np.sort(rng.uniform(-3.0, 3.0, n_tr))
    y_tr = target(x_tr) + rng.normal(0, 0.18, n_tr)
    x_te = np.linspace(-3.0, 3.0, n_te)
    y_te = target(x_te)  # clean test target: measure recovery of the true signal

    Xtr = x_tr[:, None]
    Xte = x_te[:, None]
    sig = 0.45
    Ktr = rbf(Xtr, Xtr, sig)
    Kte = rbf(Xte, Xtr, sig)  # test-to-train, for prediction

    # The price list: eigen-spectrum of the (symmetric PSD) train Gram matrix.
    eigs = np.linalg.eigvalsh(Ktr)[::-1]  # descending
    eigs = np.clip(eigs, 0, None)

    # For the points<->modes bridge we need the eigenvectors too. The discrete
    # Mercer decomposition K_ij = sum_k lambda_k u_k[i] u_k[j] gives the empirical
    # modes phi_k(x_i) = u_k[i] (unit eigenvectors of K). A bump's shopping list is
    # k(x_i, .) = sum_k lambda_k phi_k(x_i) phi_k, so a representer sum
    # f = sum_i alpha_i k(x_i, .) buys mode k in amount
    #   c_k = lambda_k * sum_i alpha_i phi_k(x_i) = lambda_k * (u_k . alpha),
    # and the two currencies agree exactly:
    #   sum_k c_k^2 / lambda_k = sum_k lambda_k (u_k . alpha)^2 = alpha^T K alpha.
    evals, evecs = np.linalg.eigh(Ktr)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]

    return dict(
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        Ktr=Ktr, Kte=Kte, eigs=eigs, evals=evals, evecs=evecs, sig=sig,
    )


def sweep_regression(D, lambdas):
    Ktr, Kte = D["Ktr"], D["Kte"]
    y_tr, y_te = D["y_tr"], D["y_te"]
    eigs = D["eigs"]
    rows = []
    for lam in lambdas:
        alpha = kernel_ridge(Ktr, y_tr, lam)
        pred_tr = Ktr @ alpha
        pred_te = Kte @ alpha
        train_mse = float(np.mean((pred_tr - y_tr) ** 2))
        test_mse = float(np.mean((pred_te - y_te) ** 2))
        rows.append(dict(
            lam=float(lam),
            norm_sq=rkhs_norm_sq(Ktr, alpha),
            d_eff=eff_dim(eigs, lam),
            train_mse=train_mse,
            test_mse=test_mse,
        ))
    return rows


def mode_bridge(D, lam):
    """c_k = lambda_k (u_k . alpha): the same weight priced by modes.

    Verifies the exchange rate against the direct RKHS bill:
        sum_k c_k^2 / lambda_k  ==  alpha^T K alpha.
    """
    Ktr = D["Ktr"]
    alpha = kernel_ridge(Ktr, D["y_tr"], lam)
    evals, evecs = D["evals"], D["evecs"]
    keep = evals > 1e-9
    ev = evals[keep]
    U = evecs[:, keep]
    proj = U.T @ alpha                       # u_k . alpha
    c = ev * proj                            # c_k = lambda_k (u_k . alpha)
    bill_modes = float(np.sum(c ** 2 / ev))  # sum_k c_k^2 / lambda_k
    bill_direct = rkhs_norm_sq(Ktr, alpha)
    return c, ev, bill_modes, bill_direct


# --------------------------------------------------------------------------- #
# Dataset B: the six-point, two-class miniature from the explainer. Carries the
# per-point |alpha_i| graded shrink and the assembly order.
# --------------------------------------------------------------------------- #
def build_miniature():
    # Two classes, three points each, laid out so a couple of points are
    # irreplaceable (alone on the boundary) and a couple are redundant (in a
    # crowd of same-class neighbors).
    X = np.array([
        [-1.6, 0.7], [-1.2, 1.0], [-0.2, 0.9],   # class +1 (upper-left arc)
        [1.5, -0.6], [1.1, -1.0], [0.1, -0.9],   # class -1 (lower-right arc)
    ])
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    sig = 0.9
    K = rbf(X, X, sig)
    eigs = np.clip(np.linalg.eigvalsh(K)[::-1], 0, None)
    return dict(X=X, y=y, K=K, eigs=eigs, sig=sig)


def sweep_miniature(M, lambdas):
    K, y, eigs = M["K"], M["y"], M["eigs"]
    rows = []
    for lam in lambdas:
        alpha = kernel_ridge(K, y, lam)
        rows.append(dict(
            lam=float(lam),
            alpha=[float(a) for a in alpha],
            abs_alpha=[float(abs(a)) for a in alpha],
            norm_sq=rkhs_norm_sq(K, alpha),
            d_eff=eff_dim(eigs, lam),
        ))
    return rows


def field_grid(M, lam, res=120):
    """The decision field f(x) = sum_i alpha_i k(x_i, x) on a grid, for GIF (a)."""
    K = M["K"]
    alpha = kernel_ridge(K, M["y"], lam)
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, res), np.linspace(-2.0, 2.0, res))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    Kg = rbf(G, M["X"], M["sig"])
    field = (Kg @ alpha).reshape(res, res)
    return field


# --------------------------------------------------------------------------- #
# Run it all and export.
# --------------------------------------------------------------------------- #
def main():
    lambdas = np.geomspace(1e-3, 1e2, 60)

    D = build_regression()
    reg_rows = sweep_regression(D, lambdas)

    # The generalization sweet spot: lambda minimizing test MSE.
    test = np.array([r["test_mse"] for r in reg_rows])
    star = int(np.argmin(test))
    lam_star = reg_rows[star]["lam"]

    # The mode bridge at a mid budget, with the exchange-rate check.
    c_mid, ev_mid, bill_modes, bill_direct = mode_bridge(D, lam=lambdas[len(lambdas) // 3])

    M = build_miniature()
    mini_rows = sweep_miniature(M, lambdas)

    # Prediction curves at three budgets for the "knob turning" GIF.
    fit_lams = [reg_rows[3]["lam"], lam_star, reg_rows[-8]["lam"]]
    grid_x = np.linspace(-3.0, 3.0, 300)
    fits = []
    for lam in fit_lams:
        alpha = kernel_ridge(D["Ktr"], D["y_tr"], lam)
        Kq = rbf(grid_x[:, None], D["x_tr"][:, None], D["sig"])
        fits.append(dict(lam=float(lam), pred=[float(v) for v in (Kq @ alpha)]))

    export = dict(
        meta=dict(
            kernel="gaussian_rbf",
            sig_reg=D["sig"], sig_mini=M["sig"],
            n_train=int(len(D["x_tr"])), n_test=int(len(D["x_te"])),
            lambdas=[float(l) for l in lambdas],
            solve="analytic (K + lambda I) alpha = y, no gradient descent",
        ),
        regression=dict(
            x_tr=[float(v) for v in D["x_tr"]],
            y_tr=[float(v) for v in D["y_tr"]],
            x_te=[float(v) for v in D["x_te"]],
            y_te=[float(v) for v in D["y_te"]],
            grid_x=[float(v) for v in grid_x],
            fits=fits,
            eigs=[float(v) for v in D["eigs"]],
            sweep=reg_rows,
            lam_star=float(lam_star),
            star_index=star,
            test_mse_star=float(test[star]),
        ),
        modes=dict(
            lam=float(lambdas[len(lambdas) // 3]),
            eigs=[float(v) for v in ev_mid],
            c_k=[float(v) for v in c_mid],
            survival=[float(e / (e + lambdas[len(lambdas) // 3])) for e in ev_mid],
            bill_from_modes=bill_modes,
            bill_direct=bill_direct,
        ),
        miniature=dict(
            X=[[float(a), float(b)] for a, b in M["X"]],
            y=[float(v) for v in M["y"]],
            eigs=[float(v) for v in M["eigs"]],
            sweep=mini_rows,
            fields=dict(),  # filled below at three budgets
        ),
    )

    # Decision fields at loose / mid / tight budgets for the knob GIF.
    for tag, lam in [("loose", lambdas[3]), ("mid", lambdas[28]), ("tight", lambdas[-10])]:
        f = field_grid(M, lam, res=90)
        export["miniature"]["fields"][tag] = dict(
            lam=float(lam), field=[[float(v) for v in row] for row in f]
        )

    (OUT / "pricelist.json").write_text(json.dumps(export))

    print("PRICELIST_EXPORT_DONE")
    print(f"  wrote {OUT/'pricelist.json'} ({(OUT/'pricelist.json').stat().st_size//1024} KB)")
    print(f"  train n={len(D['x_tr'])}, test n={len(D['x_te'])}, sig={D['sig']}")
    print(f"  lambda sweep: {lambdas[0]:.1e} .. {lambdas[-1]:.1e} ({len(lambdas)} pts)")
    print(f"  generalization sweet spot: lambda* = {lam_star:.4g}, test MSE = {test[star]:.4g}")
    print(f"  d_eff at lambda*: {reg_rows[star]['d_eff']:.2f} (of {len(D['x_tr'])} modes)")
    print(f"  d_eff at loosest lambda: {reg_rows[0]['d_eff']:.2f}; at tightest: {reg_rows[-1]['d_eff']:.2f}")
    print(f"  RKHS bill at loosest: {reg_rows[0]['norm_sq']:.3g}; at tightest: {reg_rows[-1]['norm_sq']:.3g}")
    print(f"  mode-bridge check: bill from modes {bill_modes:.5g} vs direct {bill_direct:.5g} "
          f"(rel err {abs(bill_modes-bill_direct)/bill_direct:.2e})")
    mini_first = mini_rows[0]["abs_alpha"]
    mini_last = mini_rows[-1]["abs_alpha"]
    print(f"  miniature |alpha| at loosest: {[round(v,2) for v in mini_first]}")
    print(f"  miniature |alpha| at tightest: {[round(v,3) for v in mini_last]}")
    print(f"  (none are exactly zero: min tightest |alpha| = {min(mini_last):.2e})")


if __name__ == "__main__":
    main()
