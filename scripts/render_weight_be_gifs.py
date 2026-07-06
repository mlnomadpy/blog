"""Faithful GIFs for the "What Can a Weight Be?" JAX companion.

Every moving thing is a real number computed from the kernel math (eigenvalues by
FFT, RKHS norms, kernel-ridge solves, Legendre spectra). No synthetic metaphors.

  1. weight-be-spectra.png  (static figure) the eigenvalue decay of three kernels:
                            Gaussian (a cliff), Sobolev/Laplace (a gentle slope),
                            and the Yat denominator / inverse-multiquadric (an
                            exponential, sitting between the two).
  2. weight-be-bill.png     (static figure) a corner's RKHS bill Σ_{k≤K} cₖ²/λₖ: it
                            plateaus (finite, admitted) only under the Sobolev kernel;
                            under the Gaussian and the IMQ it runs to infinity.
  3. weight-be-corner.gif   kernel ridge lowering its penalty: only the Sobolev
                            (Laplace) fit finds the corner; the Gaussian rounds it.
  4. weight-be-sphere.gif   on the sphere, sharpening a zonal kernel lets the same
                            weight ripple finer (real Legendre spectrum + a real
                            sample weight on a shaded globe).

Run: python scripts/render_weight_be_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import os
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; PUB = ROOT / 'public'
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
GAUSS_C = '#4a7fb3'; SOB_C = '#7bbf5a'; YAT_C = '#e0a45a'; RED = '#c2553a'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})


def ease(t): return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw(); a = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return a


def save_gif(path, frames, fps, hold=14):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    imageio.mimsave(path, frames + [frames[-1]] * hold, duration=1 / fps, loop=0, palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


def save_png(path, fig):
    fig.savefig(path, dpi=118, facecolor=BG)
    plt.close(fig)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# ── kernel eigenvalues on the circle = the real DFT of the kernel (jax) ──
N = 2048
GRID = jnp.linspace(-jnp.pi, jnp.pi, N, endpoint=False)
@jax.jit
def spectrum(kd):                                      # λ_k, real, normalised to λ_1 = 1
    lam = jnp.fft.rfft(jnp.fft.ifftshift(kd)).real
    lam = jnp.clip(lam, 1e-30, None)
    return lam / lam[1]
K = 40
SIG = 0.35; EPS = 0.12
# the eigenvalues are the kernel's spectral density; for these kernels it is known
# in closed form (the FFT above agrees down to float precision, then floors out).
# Gaussian: super-exponential.  Laplace/Sobolev: polynomial ~k⁻².  IMQ: exponential.
def lam_of(name, k):                                   # λ_k spectral density, normalised λ_1 = 1
    k = np.asarray(k, float)
    if name == 'Gaussian': return np.exp(-0.5 * (SIG * k) ** 2) / np.exp(-0.5 * SIG ** 2)
    if name == 'Sobolev':  return (1 + SIG ** 2) / (1 + (SIG * k) ** 2)
    return np.exp(-np.sqrt(EPS) * (k - 1))             # Yat / IMQ: exponential
def ck2_of(k):                                         # corner (triangle): cₖ² ~ 1/k⁴ on odd k
    k = np.asarray(k, float); return np.where(np.mod(k, 2) == 1, (1.0 / k ** 2) ** 2, 0.0)
_k = np.arange(K + 1)
LAM = {n: lam_of(n, _k) for n in ('Gaussian', 'Sobolev', 'Yat')}


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — the three eigenvalue spectra (static). Three fixed decay curves;
# the whole story is the final plot, so this is a still figure, not an animation.
# ═══════════════════════════════════════════════════════════════════════════
def png_spectra():
    ks = np.arange(1, K + 1)
    series = [('Gaussian (a cliff)', LAM['Gaussian'], GAUSS_C),
              ('Sobolev / Matérn (gentle, ~k⁻²)', LAM['Sobolev'], SOB_C),
              ('Yat denominator / IMQ (exponential)', LAM['Yat'], YAT_C)]
    fig = plt.figure(figsize=(6.6, 5.0), dpi=118, facecolor=BG)
    fig.text(0.5, 0.95, 'A kernel is a price list: how fast its eigenvalues fall', ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.905, 'λₖ on a log scale (tall = cheap). The decay is the whole personality of the kernel.', ha='center', fontsize=9, color=MUTED)
    ax = fig.add_axes([0.12, 0.12, 0.84, 0.74]); ax.set_facecolor(PANEL)
    for name, lam, col in series:
        ax.plot(ks, np.log10(np.maximum(lam[1:K + 1], 1e-16)), color=col, lw=2.4)
        ax.scatter([ks[-1]], [np.log10(max(lam[K], 1e-16))], s=30, color=col, zorder=5, edgecolors=BG)
        ax.text(0.02, {'Gaussian (a cliff)': 0.16, 'Sobolev / Matérn (gentle, ~k⁻²)': 0.09, 'Yat denominator / IMQ (exponential)': 0.02}[name],
                name, transform=ax.transAxes, color=col, fontsize=9.5, weight='bold')
    ax.set_xlim(1, K); ax.set_ylim(-15, 0.5)
    ax.set_xlabel('frequency mode k', color=MUTED, fontsize=10)
    ax.set_ylabel('log₁₀ λₖ', color=MUTED, fontsize=10); ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    save_png(PUB / 'weight-be-spectra.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — a corner's RKHS bill (static). Three fixed cumulative-sum curves;
# the final plot carries the whole comparison, so this is a still figure.
# ═══════════════════════════════════════════════════════════════════════════
def png_bill():
    KB = 90; ks = np.arange(1, KB + 1)
    c = ck2_of(ks); c = c / c.sum()                    # unit-energy corner
    bills = {n: np.cumsum(c / np.maximum(lam_of(n, ks), 1e-30)) for n in ('Gaussian', 'Sobolev', 'Yat')}
    cols = {'Gaussian': GAUSS_C, 'Sobolev': SOB_C, 'Yat': YAT_C}
    labels = {'Gaussian': 'Gaussian → ∞ (fast)', 'Sobolev': 'Sobolev: finite, admitted', 'Yat': 'Yat / IMQ → ∞ (slowly)'}
    CAP = 1e6
    fig = plt.figure(figsize=(6.6, 5.0), dpi=118, facecolor=BG)
    fig.text(0.5, 0.95, 'A corner’s bill, mode by mode:  ‖f‖² = Σ cₖ²/λₖ', ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.905, 'only the Sobolev bill stays finite; a corner is too rough for the Gaussian and for Yat', ha='center', fontsize=8.8, color=MUTED)
    ax = fig.add_axes([0.13, 0.12, 0.83, 0.74]); ax.set_facecolor(PANEL)
    for name in ['Sobolev', 'Yat', 'Gaussian']:
        run = np.minimum(bills[name], CAP)
        ax.plot(ks, np.log10(np.maximum(run, 1e-2)), color=cols[name], lw=2.4)
        yv = np.log10(max(min(bills[name][-1], CAP), 1e-2))
        ax.scatter([ks[-1]], [yv], s=30, color=cols[name], zorder=5, edgecolors=BG)
    for i, name in enumerate(['Gaussian', 'Yat', 'Sobolev']):
        ax.text(0.02, 0.93 - i * 0.07, labels[name], transform=ax.transAxes, color=cols[name], fontsize=9.5, weight='bold')
    ax.axhline(np.log10(CAP), color=RED, lw=0.8, ls=':')
    ax.text(KB, np.log10(CAP), ' unaffordable', color=RED, fontsize=8, va='bottom', ha='right')
    ax.set_xlim(1, KB); ax.set_ylim(-2, 6.4)
    ax.set_xlabel('modes summed, up to k', color=MUTED, fontsize=10)
    ax.set_ylabel('log₁₀ running bill', color=MUTED, fontsize=10); ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    save_png(PUB / 'weight-be-bill.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — kernel ridge corner fit: only Sobolev finds the corner
# ═══════════════════════════════════════════════════════════════════════════
def gif_corner():
    n = 23; X = np.linspace(-1.05, 1.05, n); Y = 1 - np.abs(X)        # a tent, point pinned at 0
    Xj = jnp.asarray(X)
    def fit(kfn, lam):
        G = kfn(Xj[:, None], Xj[None, :]); a = jnp.linalg.solve(G + lam * jnp.eye(n), jnp.asarray(Y))
        xs = jnp.linspace(-1.05, 1.05, 240)
        return np.asarray(xs), np.asarray(kfn(xs[:, None], Xj[None, :]) @ a)
    kG = lambda a, b: jnp.exp(-((a - b) ** 2) / (2 * 0.14 ** 2))
    kL = lambda a, b: jnp.exp(-jnp.abs(a - b) / 0.14)
    kY = lambda a, b: 1.0 / ((a - b) ** 2 + 0.03)                    # Yat / inverse-multiquadric gate
    lams = np.geomspace(0.5, 0.004, 30)                              # lower the penalty over the animation
    frames = []
    for fi, lam in enumerate(lams):
        xg, yG = fit(kG, lam); _, yL = fit(kL, lam); _, yY = fit(kY, lam)
        fig = plt.figure(figsize=(6.4, 5.0), dpi=116, facecolor=BG)
        fig.text(0.5, 0.95, 'Lowering the penalty, only the Sobolev fit finds the corner', ha='center', fontsize=13, weight='bold')
        fig.text(0.5, 0.905, 'the same tent, fit by kernel ridge with three kernels; both smooth ones round the corner', ha='center', fontsize=8.8, color=MUTED)
        ax = fig.add_axes([0.06, 0.08, 0.9, 0.80]); ax.set_facecolor(PANEL)
        ax.plot([-1, 0, 1], [0, 1, 0], color=MUTED, lw=1.1, ls='--')
        ax.plot(xg, yG, color=GAUSS_C, lw=2.4); ax.plot(xg, yY, color=YAT_C, lw=2.4); ax.plot(xg, yL, color=SOB_C, lw=2.4)
        ax.scatter(X, Y, s=14, color=MUTED, alpha=0.8, zorder=3)
        ax.text(0.03, 0.92, 'Gaussian: rounds it', transform=ax.transAxes, color=GAUSS_C, fontsize=10, weight='bold')
        ax.text(0.03, 0.85, 'Yat / IMQ: rounds it too', transform=ax.transAxes, color=YAT_C, fontsize=10, weight='bold')
        ax.text(0.03, 0.78, 'Sobolev H¹: turns it', transform=ax.transAxes, color=SOB_C, fontsize=10, weight='bold')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.12, 1.18)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.text(0.97, 0.92, f'penalty λ = {lam:.3f}', transform=ax.transAxes, ha='right', fontsize=10, color=INK, weight='bold')
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'weight-be-corner.gif', frames, fps=13, hold=16)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 — the sphere: sharpening a zonal kernel lets the weight ripple finer
# ═══════════════════════════════════════════════════════════════════════════
def legendre(k, t):
    p0 = np.ones_like(t); p1 = t
    if k == 0: return p0
    if k == 1: return p1
    for n in range(1, k):
        p2 = ((2 * n + 1) * t * p1 - n * p0) / (n + 1); p0, p1 = p1, p2
    return p1


def gif_sphere():
    KM = 10
    protos = [np.array(v) / np.linalg.norm(v) for v in
              [[0.3, 0.4, 0.86], [-0.6, -0.1, 0.79], [0.45, -0.6, 0.66]]]
    alpha = [1.0, -1.0, 1.0]
    S = 150; gx, gy = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S))
    sweeps = np.concatenate([np.linspace(2, 20, 26), np.linspace(20, 2, 10)])
    bg = np.array([22, 20, 15]); pos = np.array([224, 164, 90]); neg = np.array([74, 127, 179])
    frames = []
    for sharp in sweeps:
        # field on the front hemisphere, f(p) = Σ αⱼ exp(sharp(⟨p,Wⱼ⟩−1))
        z2 = 1 - gx ** 2 - gy ** 2; mask = z2 >= 0; zz = np.sqrt(np.clip(z2, 0, 1))
        f = np.zeros_like(gx)
        for a, W in zip(alpha, protos):
            dot = gx * W[0] + (-gy) * W[1] + zz * W[2]
            f += a * np.exp(sharp * (dot - 1))
        fmax = np.abs(f[mask]).max() + 1e-9; t = np.clip(f / fmax, -1, 1)
        img = np.zeros((S, S, 3), np.uint8)
        shade = (0.55 + 0.45 * zz)[..., None]
        col = np.where(t[..., None] >= 0, bg + (pos - bg) * np.abs(t)[..., None], bg + (neg - bg) * np.abs(t)[..., None])
        img[mask] = (col * shade)[mask].astype(np.uint8)
        # real Legendre spectrum bₖ of κ(τ)=exp(sharp(τ−1))
        tt = np.linspace(-1, 1, 400); kap = np.exp(sharp * (tt - 1))
        bk = np.array([(2 * k + 1) / 2 * np.trapezoid(kap * legendre(k, tt), tt) for k in range(KM + 1)])
        bk = np.maximum(bk / bk[0], 1e-4)
        fig = plt.figure(figsize=(7.6, 4.4), dpi=112, facecolor=BG)
        fig.text(0.5, 0.94, 'On the sphere, sharpening the kernel lets the weight ripple finer', ha='center', fontsize=13, weight='bold')
        fig.text(0.5, 0.875, 'a real weight f(p) = Σ αⱼ κ(⟨p, Wⱼ⟩) on the globe, and the kernel’s real Legendre spectrum bₖ', ha='center', fontsize=8.4, color=MUTED)
        axg = fig.add_axes([0.02, 0.04, 0.46, 0.78]); axg.imshow(img, origin='upper'); axg.set_xticks([]); axg.set_yticks([])
        for s in axg.spines.values(): s.set_color(LINE)
        axg.set_title('the weight on the sphere', fontsize=9.5, color=MUTED, pad=3)
        axb = fig.add_axes([0.58, 0.16, 0.39, 0.6]); axb.set_facecolor(PANEL)
        axb.bar(range(KM + 1), bk, color=YAT_C, width=0.72)
        axb.set_yscale('log'); axb.set_ylim(1e-4, 1.4); axb.set_xlabel('harmonic degree k', color=MUTED, fontsize=9)
        axb.set_title(f'spectrum bₖ   (sharpness {sharp:.0f})', fontsize=9.5, color=INK, pad=3)
        axb.tick_params(colors=MUTED, labelsize=8)
        for s in axb.spines.values(): s.set_color(LINE)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'weight-be-sphere.gif', frames, fps=13, hold=10)


if __name__ == '__main__':
    print('decay check (λ_k at k=8,16,32):')
    for n in ['Gaussian', 'Sobolev', 'Yat']:
        print(f'  {n:9s}', [f'{LAM[n][k]:.1e}' for k in (8, 16, 32)])
    png_spectra(); png_bill(); gif_corner(); gif_sphere()
    print('WEIGHT_BE_GIFS_DONE')
