"""
mo_nn_solver.py - neural-network wavefunction solver (variational Monte Carlo, JAX)
==================================================================================
Finds the ground state of one or two electrons around the two nuclei of the
MO Electron Simulator (same model: 2D, soft-core Coulomb interactions, fixed
nuclei) by letting a neural network *be* the wavefunction and training it to
have the lowest possible energy.

How it works
------------
  Wavefunction   psi(r1, r2) = exp( f(r1, r2) ),  f = envelope + neural network
                 * envelope: each electron decays exponentially away from the
                   nuclei (learnable decay rates), so psi is normalizable;
                 * network: per-electron features (position, distance to each
                   nucleus) -> MLP -> h_i; the two h's are combined
                   symmetrically (h1 + h2, h1 * h2) together with features of
                   the electron-electron distance, so f(r1, r2) = f(r2, r1).
                   That is the spin-singlet ground state (opposite spins), whose
                   spatial wavefunction is symmetric and has no nodes, so a
                   single exp(f) can represent it exactly.
  Energy         E = < E_L >  averaged over positions sampled from |psi|^2,
                 local energy E_L = -1/2 (lap f + |grad f|^2) + V(r1, r2);
                 JAX computes the derivatives exactly (automatic differentiation).
  Sampling       Metropolis random walk of many "walkers" (electron configurations).
  Training       gradient dE/dtheta = 2 < (E_L - E) d f/dtheta >, Adam optimizer.
                 The variational principle guarantees E >= the exact ground state
                 energy, and E_L becomes constant (zero variance) for the exact state.

Answer keys it is checked against
---------------------------------
  1 electron   the simulator's grid solver (essentially exact).
  2 electrons  * mean field (Hartree, both electrons in one shared sigma orbital,
                 what the simulator's "Relax both" does): an upper bound that
                 misses electron correlation;
               * the exact solution on a full 4D grid (x1, y1, x2, y2), affordable
                 here because the model is 2D with only two electrons. This is
                 what the network should reach, and what becomes impossible
                 for more electrons or 3D - where the network approach still works.

Run
---
    python mo_nn_solver.py                       H2 (2 electrons), R = 2 bohr
    python mo_nn_solver.py --electrons 1         H2+ (1 electron)
    python mo_nn_solver.py --R 3.0               stretched bond (mean field gets worse)
    python mo_nn_solver.py --curve 1.0,1.5,2,3,4 energy vs bond length, all methods
    python mo_nn_solver.py --selftest            checks of the solver itself
  options: --iterations N  --walkers N  --hidden N  --seed N  --Z1 --Z2 --soft
           --out DIR (default nn_results)  --no-exact  --fp32
Results (table + plots) are printed and saved as PNG files in the output folder.

Setup (once):
    python -m pip install numpy matplotlib jax
    (NVIDIA GPU: python -m pip install "jax[cuda12]"; then --fp32 is recommended)

Units: atomic units (bohr, hartree). Energies are electronic (nuclear repulsion
Z1*Z2/R is added only where a "total" energy is shown).
"""

import argparse
import math
import os
import sys
import time

if "--fp32" in sys.argv:                       # must be decided before JAX is imported
    os.environ["MO_JAX_FP32"] = "1"

import numpy as np                             # noqa: E402
import jax                                     # noqa: E402

FP32 = os.environ.get("MO_JAX_FP32", "") not in ("", "0")
jax.config.update("jax_enable_x64", not FP32)
import jax.numpy as jnp                        # noqa: E402
from jax import lax                            # noqa: E402

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DEFAULTS = dict(R=2.0, Z1=1.0, Z2=1.0, soft=0.5)   # same molecule as the simulator
HIDDEN = 32               # width of the network layers
DEPTH = 2                 # hidden layers per network block
WALKERS = 2048            # electron configurations sampled in parallel
MCMC_STEPS = 10           # Metropolis moves per walker between parameter updates
ITERATIONS = {1: 1500, 2: 3000}
LEARNING_RATE = 3e-3
LR_DECAY = 1000           # lr = LEARNING_RATE / (1 + iteration / LR_DECAY)
CLIP_WIDTH = 5.0          # clip local energies at median +- 5 mean absolute deviations (gradient only)
EVAL_BLOCKS = 200         # evaluation: blocks of MCMC_STEPS moves, one energy average per block
EXACT_GRID_N = 52         # 4D reference grid points per axis (52^4 = 7.3 million, dx ~0.3 bohr;
                          # discretization error ~2e-5 Ha, about 2 minutes on a 4-core CPU)


# ============================================================================
# The model: molecule, potential, network wavefunction
# ============================================================================
def molecule(R, Z1, Z2, soft):
    """Molecule as a dict of arrays (a JAX "pytree"), so changing R does not recompile."""
    return dict(xn=jnp.array([-R / 2, R / 2]), Z=jnp.array([Z1, Z2]), a2=jnp.asarray(soft ** 2))


def potential(r, mol):
    """r: (n_el, 2). Soft-core electron-nucleus attraction + electron-electron repulsion."""
    d2 = (r[:, 0:1] - mol["xn"][None, :]) ** 2 + r[:, 1:2] ** 2 + mol["a2"]   # (n_el, 2)
    v = -jnp.sum(mol["Z"][None, :] / jnp.sqrt(d2))
    if r.shape[0] == 2:
        v = v + 1.0 / jnp.sqrt(jnp.sum((r[0] - r[1]) ** 2) + mol["a2"])
    return v


def _mlp_init(key, sizes, last_scale=1.0):
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(keys[k], (m, n)) / math.sqrt(m)
        if k == len(sizes) - 2:
            w = w * last_scale
        layers.append((w, jnp.zeros(n)))
    return layers


def _mlp(layers, x, linear_out):
    for k, (w, b) in enumerate(layers):
        x = x @ w + b
        if not (linear_out and k == len(layers) - 1):
            x = jnp.tanh(x)
    return x


def init_params(key, n_el, hidden=HIDDEN, depth=DEPTH):
    k = jax.random.split(key, 3)
    head_in = hidden if n_el == 1 else 3 * hidden
    params = dict(
        env=dict(log_kappa=jnp.zeros(2), w=jnp.zeros(2)),
        phi=_mlp_init(k[0], [4] + [hidden] * depth),                 # per-electron features
        head=_mlp_init(k[1], [head_in, hidden, 1], last_scale=0.01),  # starts ~ pure envelope
    )
    if n_el == 2:
        params["pair"] = _mlp_init(k[2], [2] + [hidden] * depth)     # electron-electron features
    return params


def n_params(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def log_psi(params, r, mol):
    """log psi for one configuration r: (n_el, 2)."""
    d = jnp.sqrt((r[:, 0:1] - mol["xn"][None, :]) ** 2 + r[:, 1:2] ** 2 + 1.0)    # (n_el, 2)
    kappa = jnp.exp(params["env"]["log_kappa"]) * mol["Z"]
    envelope = jnp.sum(jax.nn.logsumexp(params["env"]["w"][None, :] - kappa[None, :] * d, axis=1))
    h = _mlp(params["phi"], jnp.concatenate([r, d], axis=1), linear_out=False)   # (n_el, hidden)
    if r.shape[0] == 1:
        x = h[0]
    else:
        r12 = jnp.sqrt(jnp.sum((r[0] - r[1]) ** 2) + mol["a2"])
        g = _mlp(params["pair"], jnp.stack([r12, 1.0 / r12]), linear_out=False)
        x = jnp.concatenate([h[0] + h[1], h[0] * h[1], g])           # symmetric in the electrons
    return envelope + _mlp(params["head"], x, linear_out=True)[0]


def local_energy(params, r, mol):
    """E_L = (H psi) / psi = -1/2 (lap log psi + |grad log psi|^2) + V."""
    shape = r.shape
    f = lambda flat: log_psi(params, flat.reshape(shape), mol)       # noqa: E731
    flat = r.reshape(-1)
    g = jax.grad(f)(flat)
    lap = jnp.trace(jax.hessian(f)(flat))
    return -0.5 * (lap + g @ g) + potential(r, mol)


_batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0, None))
_batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, None))


# ============================================================================
# Sampling and training
# ============================================================================
def _mcmc(params, mol, walkers, logp, key, step, n_steps):
    """Metropolis random walk in |psi|^2 (logp = 2 log psi). Returns walkers, logp, acceptance."""
    def body(_, carry):
        w, lp, key, acc = carry
        key, k1, k2 = jax.random.split(key, 3)
        prop = w + step * jax.random.normal(k1, w.shape, w.dtype)
        lp_new = 2.0 * _batch_log_psi(params, prop, mol)
        accept = jnp.log(jax.random.uniform(k2, lp.shape, w.dtype)) < lp_new - lp
        w = jnp.where(accept[:, None, None], prop, w)
        lp = jnp.where(accept, lp_new, lp)
        return w, lp, key, acc + jnp.mean(accept)
    w, lp, _, acc = lax.fori_loop(0, n_steps, body, (walkers, logp, key, 0.0))
    return w, lp, acc / n_steps


_mcmc_jit = jax.jit(_mcmc, static_argnames=("n_steps",))


def _adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=z, v=z, t=jnp.zeros((), jnp.int32))


def _adam(params, grads, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, s["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, s["v"], grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps),
                                    params, m, v)
    return params, dict(m=m, v=v, t=t)


@jax.jit
def _train_step(params, opt, walkers, logp, key, step, lr, mol):
    key, k = jax.random.split(key)
    walkers, logp, acc = _mcmc(params, mol, walkers, logp, k, step, MCMC_STEPS)
    e_l = _batch_local_energy(params, walkers, mol)
    med = jnp.median(e_l)
    mad = jnp.mean(jnp.abs(e_l - med))
    e_c = jnp.clip(e_l, med - CLIP_WIDTH * mad, med + CLIP_WIDTH * mad)

    def surrogate(p):                        # its gradient is 2 < (E_L - E) d log psi >
        return 2.0 * jnp.mean(lax.stop_gradient(e_c - jnp.mean(e_c)) * _batch_log_psi(p, walkers, mol))

    grads = jax.grad(surrogate)(params)
    params, opt = _adam(params, grads, opt, lr)
    logp = 2.0 * _batch_log_psi(params, walkers, mol)
    return params, opt, walkers, logp, key, jnp.mean(e_l), jnp.var(e_l), acc


def init_walkers(key, n_walkers, n_el, mol):
    k1, k2 = jax.random.split(key)
    centre = jax.random.randint(k1, (n_walkers, n_el), 0, 2)
    start = jnp.stack([mol["xn"][centre], jnp.zeros((n_walkers, n_el))], axis=-1)
    return start + jax.random.normal(k2, (n_walkers, n_el, 2))


def train(mol, n_el, iterations, n_walkers=WALKERS, hidden=HIDDEN, seed=0, params=None, log_every=100,
          reference=None, quiet=False):
    """Train the network wavefunction. Returns a state dict (params, walkers, history...)."""
    key = jax.random.PRNGKey(seed)
    key, kp, kw = jax.random.split(key, 3)
    if params is None:
        params = init_params(kp, n_el, hidden)
    opt = _adam_init(params)
    walkers = init_walkers(kw, n_walkers, n_el, mol)
    logp = 2.0 * _batch_log_psi(params, walkers, mol)
    step = 0.5
    for _ in range(4):                                   # burn-in with step-size tuning
        key, k = jax.random.split(key)
        walkers, logp, acc = _mcmc_jit(params, mol, walkers, logp, k, step, 50)
        step *= float(np.exp(float(acc) - 0.5))
    hist = []
    t0 = time.time()
    if not quiet:
        print(f"  training: {n_el} electron(s), {n_params(params)} network parameters, "
              f"{n_walkers} walkers, {iterations} iterations on {jax.devices()[0].platform.upper()}"
              f" ({'32' if FP32 else '64'}-bit)", flush=True)
    for it in range(iterations):
        lr = LEARNING_RATE / (1 + it / LR_DECAY)
        params, opt, walkers, logp, key, e, var, acc = _train_step(params, opt, walkers, logp, key,
                                                                  step, lr, mol)
        acc = float(acc)
        step = float(np.clip(step * np.exp(acc - 0.5), 0.02, 3.0))   # keep acceptance near 50 %
        hist.append((float(e), float(var), acc))
        if not np.isfinite(hist[-1][0]):
            raise FloatingPointError(f"energy became {hist[-1][0]} at iteration {it}")
        if not quiet and (it + 1) % log_every == 0:
            recent = np.array(hist[-log_every:])
            ref = f"  (reference {reference:.5f})" if reference is not None else ""
            print(f"  iter {it + 1:5d}  E = {recent[:, 0].mean():.5f} Ha  var(E_L) = {recent[:, 1].mean():.2e}"
                  f"  acceptance {recent[:, 2].mean():.2f}  [{time.time() - t0:.0f} s]{ref}", flush=True)
    return dict(params=params, walkers=walkers, logp=logp, key=key, step=step,
                history=np.array(hist), seconds=time.time() - t0, mol=mol, n_el=n_el)


@jax.jit
def _eval_block(params, walkers, logp, key, step, mol, edges):
    key, k = jax.random.split(key)
    walkers, logp, acc = _mcmc(params, mol, walkers, logp, k, step, MCMC_STEPS)
    e_l = _batch_local_energy(params, walkers, mol)
    pts = walkers.reshape(-1, 2)
    hist, _, _ = jnp.histogram2d(pts[:, 0], pts[:, 1], bins=(edges, edges))
    return walkers, logp, key, jnp.mean(e_l), jnp.var(e_l), hist


def evaluate(state, blocks=EVAL_BLOCKS, half_width=6.0, bins=96):
    """Frozen network: energy with error bar, E_L variance and the electron density."""
    edges = jnp.linspace(-half_width, half_width, bins + 1)
    w, lp, key = state["walkers"], state["logp"], state["key"]
    means, variances, dens = [], [], 0.0
    for _ in range(blocks):
        w, lp, key, e, v, h = _eval_block(state["params"], w, lp, key, state["step"], state["mol"], edges)
        means.append(float(e))
        variances.append(float(v))
        dens = dens + h
    means = np.array(means)
    dx = float(edges[1] - edges[0])
    n_samples = blocks * w.shape[0]
    density = np.asarray(dens).T / (n_samples * dx * dx)       # [y, x], integrates to n_el
    return dict(E=float(means.mean()), err=float(means.std(ddof=1) / np.sqrt(blocks)),
                var=float(np.mean(variances)), density=density, edges=np.asarray(edges))


# ============================================================================
# Answer keys (grid solvers)
# ============================================================================
def grid_one_electron(R, Z1, Z2, soft):
    """sigma orbital on the simulator's fine 2D grid. Returns (E, x, density[y, x])."""
    import mo_physics as m
    s = m.TwoCenterSystem(R, Z1, Z2, soft)
    q = m.QuantumElectron(s, absorb=False)
    psi, e = q.solve_state(q.lcao(+1), tol=1e-11)
    return e, s.x, np.abs(np.asarray(psi)) ** 2


def grid_mean_field(R, Z1, Z2, soft):
    """Restricted mean field (both electrons in one shared orbital), as in the simulator.
    Returns (E, x, density[y, x])."""
    import mo_physics as m
    s = m.TwoCenterSystem(R, Z1, Z2, soft)
    sim = m.Simulation(s, absorb=False)
    sim.count = 2
    q = sim.electrons[0].wave
    for e in sim.electrons:
        e.mode = "wave"
        e.wave.psi = q.lcao(+1)
    e_old, e = np.inf, sim.observe()["E"]
    steps = 0
    while abs(e - e_old) > 1e-10 and steps < 8000:
        sim.relax_shared(20)
        steps += 20
        e_old, e = e, sim.observe()["E"]
    return e, s.x, 2 * np.abs(np.asarray(q.psi)) ** 2


def grid_exact_two_electron(R, Z1, Z2, soft, n=EXACT_GRID_N, box=None, tol=1e-10, verbose=True):
    """Exact two-electron (singlet) ground state on a 4D grid psi(x1, y1, x2, y2), by
    imaginary-time evolution (split operator, real FFTs). Returns (E, x, density[y, x])."""
    box = box or max(15.5, R + 11.5)
    dx = box / n
    x = (np.arange(n) - n / 2 + 0.5) * dx
    a2 = soft ** 2
    X1 = jnp.asarray(x)[:, None, None, None]
    Y1 = jnp.asarray(x)[None, :, None, None]
    X2 = jnp.asarray(x)[None, None, :, None]
    Y2 = jnp.asarray(x)[None, None, None, :]

    def v1(xx, yy):
        return -Z1 / jnp.sqrt((xx + R / 2) ** 2 + yy ** 2 + a2) - Z2 / jnp.sqrt((xx - R / 2) ** 2 + yy ** 2 + a2)

    V = v1(X1, Y1) + v1(X2, Y2) + 1.0 / jnp.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2 + a2)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    kr = 2 * np.pi * np.fft.rfftfreq(n, dx)
    K = 0.5 * (jnp.asarray(k)[:, None, None, None] ** 2 + jnp.asarray(k)[None, :, None, None] ** 2
               + jnp.asarray(k)[None, None, :, None] ** 2 + jnp.asarray(kr)[None, None, None, :] ** 2)
    orb = jnp.exp(-jnp.sqrt((X1 + R / 2) ** 2 + Y1 ** 2)) + jnp.exp(-jnp.sqrt((X1 - R / 2) ** 2 + Y1 ** 2))
    orb2 = jnp.exp(-jnp.sqrt((X2 + R / 2) ** 2 + Y2 ** 2)) + jnp.exp(-jnp.sqrt((X2 - R / 2) ** 2 + Y2 ** 2))
    psi = orb * orb2                                          # symmetric start: singlet
    psi = psi / jnp.sqrt(jnp.sum(psi ** 2))

    @jax.jit
    def steps(psi, ev, ek, n_steps):
        def body(_, p):
            p = ev * p
            p = jnp.fft.irfftn(ek * jnp.fft.rfftn(p), s=p.shape)
            p = ev * p
            return p / jnp.sqrt(jnp.sum(p ** 2))
        return lax.fori_loop(0, n_steps, body, psi)

    @jax.jit
    def energy(psi):
        kin = jnp.sum(psi * jnp.fft.irfftn(K * jnp.fft.rfftn(psi), s=psi.shape))
        return (kin + jnp.sum(psi ** 2 * V)) / jnp.sum(psi ** 2)

    t0 = time.time()
    e = float(energy(psi))
    for dt, max_steps in ((0.1, 1500), (0.025, 1500)):      # coarse, then fine imaginary time step
        ev, ek = jnp.exp(-0.5 * dt * V), jnp.exp(-dt * K)
        done = 0
        while done < max_steps:
            psi = steps(psi, ev, ek, 50)
            done += 50
            e_old, e = e, float(energy(psi))
            if abs(e - e_old) < tol:
                break
    rho = 2 * np.asarray(jnp.sum(psi ** 2, axis=(2, 3))) / dx ** 2     # [x, y]; integrates to 2
    if verbose:
        print(f"  exact 4D grid: {n}^4 = {n ** 4 / 1e6:.1f}M points, dx = {dx:.3f} bohr, "
              f"E = {e:.6f} Ha  [{time.time() - t0:.0f} s]", flush=True)
    return e, x, rho.T


# ============================================================================
# Reports and plots
# ============================================================================
def _plots(out, tag, res, refs, n_el, mol_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = res["state"]["history"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    e = hist[:, 0]
    it = np.arange(1, len(e) + 1)
    win = max(1, len(e) // 50)
    smooth = np.convolve(e, np.ones(win) / win, mode="valid")
    ax.plot(it, e, color="#adb5bd", lw=0.6, label="network, each iteration")
    ax.plot(it[win - 1:], smooth, color="#1971c2", lw=1.6, label="network, running mean")
    styles = {"exact (grid)": ("black", "--"), "mean field": ("#e8590c", ":")}
    for name, (val, *_rest) in refs.items():
        col, ls = styles.get(name, ("black", "--"))
        ax.axhline(val, color=col, ls=ls, lw=1.3, label=f"{name}: {val:.4f}")
    vals = [v[0] for v in refs.values()] or [smooth.min()]
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - 0.25 * max(hi - lo, 0.02), hi + 0.6 * max(hi - lo, 0.02))
    ax.set_xlabel("training iteration")
    ax.set_ylabel("energy (hartree)")
    ax.set_title("Training: the energy goes down to the answer")
    ax.legend(fontsize=8)

    ev = res["eval"]
    ext = (ev["edges"][0], ev["edges"][-1], ev["edges"][0], ev["edges"][-1])
    vmax = ev["density"].max()
    axes[1].imshow(ev["density"], origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Electron density from the network")
    axes[1].set_xlabel("x (bohr)")
    axes[1].set_ylabel("y (bohr)")
    ref_name = "exact (grid)" if "exact (grid)" in refs else None
    ax = axes[2]
    centres = 0.5 * (ev["edges"][1:] + ev["edges"][:-1])
    mid = slice(len(centres) // 2 - 1, len(centres) // 2 + 1)
    ax.plot(centres, ev["density"][mid].mean(axis=0), color="#1971c2", lw=1.8, label="network (sampled)")
    for name in (ref_name, "mean field"):
        if name and name in refs and refs[name][2] is not None:
            _, xg, rho = refs[name]
            rows = np.argsort(np.abs(xg))[:2]             # the two grid rows straddling y = 0
            ax.plot(xg, rho[rows].mean(axis=0), color=styles[name][0], ls=styles[name][1],
                    lw=1.4, label=name)
    ax.set_xlim(ev["edges"][0], ev["edges"][-1])
    ax.set_xlabel("x along the bond (bohr)")
    ax.set_ylabel("electron density at y = 0")
    ax.set_title("Density along the bond")
    ax.legend(fontsize=8)
    R, Z1, Z2, soft = mol_args
    fig.suptitle(f"{'H2' if n_el == 2 else 'H2+'}-like model, R = {R} bohr, Z = ({Z1}, {Z2}), "
                 f"soft-core a = {soft}:  network E = {ev['E']:.5f} ± {ev['err']:.5f} Ha", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"nn_{tag}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def solve(R, Z1, Z2, soft, n_el, iterations, walkers, hidden, seed, exact=True, out=None, verbose=True):
    """Train the network for one molecule and compare it with the answer keys."""
    mol = molecule(R, Z1, Z2, soft)
    refs = {}
    if verbose:
        print(f"\n=== {'H2' if n_el == 2 else 'H2+'}-like molecule: R = {R}, Z1 = {Z1}, Z2 = {Z2}, "
              f"soft-core a = {soft} ===", flush=True)
    if n_el == 1:
        e, x, rho = grid_one_electron(R, Z1, Z2, soft)
        refs["exact (grid)"] = (e, x, rho)
    else:
        e, x, rho = grid_mean_field(R, Z1, Z2, soft)
        refs["mean field"] = (e, x, rho)
        if verbose:
            print(f"  mean field (shared orbital): E = {e:.6f} Ha", flush=True)
        if exact:
            refs["exact (grid)"] = grid_exact_two_electron(R, Z1, Z2, soft, verbose=verbose)
    ref_e = refs.get("exact (grid)", (None,))[0]
    state = train(mol, n_el, iterations, walkers, hidden, seed, reference=ref_e, quiet=not verbose)
    ev = evaluate(state)
    res = dict(state=state, eval=ev, refs={k: v[0] for k, v in refs.items()})
    if verbose:
        print(f"  network:  E = {ev['E']:.6f} ± {ev['err']:.6f} Ha   var(E_L) = {ev['var']:.2e}"
              f"   ({state['seconds']:.0f} s training)")
        for name, (val, *_r) in refs.items():
            print(f"    vs {name:13s} {val:.6f} Ha   difference {ev['E'] - val:+.6f} Ha")
        if "mean field" in refs and "exact (grid)" in refs:
            corr = refs["exact (grid)"][0] - refs["mean field"][0]
            got = ev["E"] - refs["mean field"][0]
            print(f"    correlation energy (exact - mean field) = {corr:.6f} Ha; "
                  f"the network recovers {got / corr:.1%} of it")
            print("    (the grid answer carries its own small discretization error, ~2e-5 Ha)")
    if out:
        os.makedirs(out, exist_ok=True)
        tag = f"{n_el}e_R{R:g}"
        res["plot"] = _plots(out, tag, res, refs, n_el, (R, Z1, Z2, soft))
        if verbose:
            print(f"  plot: {res['plot']}")
    return res


def bond_curve(Rs, Z1, Z2, soft, iterations, walkers, hidden, seed, out, exact=True):
    """Energy vs bond length for H2: network, exact grid, mean field (+ nuclear repulsion)."""
    rows = []
    params = None
    for R in Rs:
        mol = molecule(R, Z1, Z2, soft)
        print(f"\n--- R = {R} bohr ---", flush=True)
        e_mf = grid_mean_field(R, Z1, Z2, soft)[0]
        e_ex = grid_exact_two_electron(R, Z1, Z2, soft)[0] if exact else np.nan
        # warm start from the previous bond length: the network only has to adjust
        st = train(mol, 2, iterations if params is None else iterations // 2, walkers, hidden, seed,
                   params=params, reference=None if np.isnan(e_ex) else e_ex, log_every=500)
        params = st["params"]
        ev = evaluate(st, blocks=100)
        rep = Z1 * Z2 / R
        rows.append((R, ev["E"] + rep, ev["err"], e_ex + rep, e_mf + rep))
        print(f"  total energy incl. nuclear repulsion: network {ev['E'] + rep:.5f} ± {ev['err']:.5f}, "
              f"exact {e_ex + rep:.5f}, mean field {e_mf + rep:.5f}", flush=True)
    rows = np.array(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    if exact:
        ax.plot(rows[:, 0], rows[:, 3], "k--", lw=1.4, label="exact (4D grid)")
    ax.plot(rows[:, 0], rows[:, 4], ":", color="#e8590c", lw=1.8, marker="s", ms=4, label="mean field")
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o", color="#1971c2", ms=6,
                label="neural network (VMC)")
    ax.set_xlabel("bond length R (bohr)")
    ax.set_ylabel("total energy (hartree)")
    ax.set_title("H2 model: energy vs bond length")
    ax.legend()
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "nn_bond_curve.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("\n   R     network            exact      mean field   (total energies, Ha)")
    for R, e, err, ex, mf in rows:
        print(f"  {R:4.2f}  {e:9.5f} ± {err:.5f}  {ex:9.5f}  {mf:9.5f}")
    print(f"plot: {path}")
    return rows


# ============================================================================
# Checks of the solver itself
# ============================================================================
def run_selftest():
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    p = DEFAULTS
    mol = molecule(p["R"], p["Z1"], p["Z2"], p["soft"])
    key = jax.random.PRNGKey(1)
    # A non-trivial network: random weights, and a head that is not tiny.
    params2 = init_params(key, 2)
    params2["head"][-1] = (params2["head"][-1][0] * 30.0, params2["head"][-1][1])
    params1 = init_params(jax.random.PRNGKey(2), 1)
    params1["head"][-1] = (params1["head"][-1][0] * 30.0, params1["head"][-1][1])

    # 1. Exchange symmetry (singlet).
    r = jax.random.normal(jax.random.PRNGKey(3), (64, 2, 2)) * 1.5
    a = _batch_log_psi(params2, r, mol)
    b = _batch_log_psi(params2, r[:, ::-1], mol)
    err = float(jnp.max(jnp.abs(a - b)))
    check("swapping the electrons leaves psi unchanged", err < 1e-12, f"max |diff| = {err:.1e}")

    # 2. Kinetic energy from automatic differentiation vs finite differences.
    for n_el, prm in ((1, params1), (2, params2)):
        rr = jax.random.normal(jax.random.PRNGKey(4), (n_el, 2)) * 1.2
        f = lambda flat: log_psi(prm, flat.reshape(n_el, 2), mol)       # noqa: E731
        flat = rr.reshape(-1)
        h = 1e-4
        lap_fd = sum((f(flat + h * e) - 2 * f(flat) + f(flat - h * e)) / h ** 2 for e in jnp.eye(flat.size))
        g_fd = jnp.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in jnp.eye(flat.size)])
        el_fd = -0.5 * (lap_fd + g_fd @ g_fd) + potential(rr, mol)
        el = local_energy(prm, rr, mol)
        check(f"local energy, {n_el} electron(s): autodiff = finite differences", abs(el - el_fd) < 1e-5,
              f"{float(el):.8f} vs {float(el_fd):.8f}")

    # 3. <H> of the network wavefunction (1 electron): average of E_L over |psi|^2 on the
    #    simulator's fine grid, vs the simulator's own FFT energy of the same psi.
    import mo_physics as m
    s = m.TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
    pts = jnp.stack([jnp.asarray(s.X).ravel(), jnp.asarray(s.Y).ravel()], axis=1)[:, None, :]
    lp = np.concatenate([np.asarray(_batch_log_psi(params1, pts[i:i + 8192], mol)) for i in range(0, len(pts), 8192)])
    el = np.concatenate([np.asarray(_batch_local_energy(params1, pts[i:i + 8192], mol))
                         for i in range(0, len(pts), 8192)])
    w = np.exp(2 * (lp - lp.max()))
    e_quad = float(np.sum(w * el) / np.sum(w))
    psi_grid = np.exp(lp - lp.max()).reshape(s.n, s.n).astype(complex)
    e_fft = m.QuantumElectron(s, absorb=False).energy(psi_grid)
    check("<H> from E_L = <H> from the grid solver (same psi)", abs(e_quad - e_fft) < 1e-6,
          f"{e_quad:.8f} vs {e_fft:.8f} Ha")

    # 4. Short training of H2+ gets close to the grid answer, and stays above it (variational).
    e_grid = grid_one_electron(p["R"], p["Z1"], p["Z2"], p["soft"])[0]
    st = train(mol, 1, 600, n_walkers=1024, seed=0, quiet=True)
    ev = evaluate(st, blocks=60)
    check("short H2+ training reaches the grid energy", -3 * ev["err"] - 2e-4 < ev["E"] - e_grid < 5e-3,
          f"network {ev['E']:.5f} ± {ev['err']:.5f}, grid {e_grid:.5f} Ha, var(E_L) {ev['var']:.1e}")

    # 5. The exact 4D solver: the electron-electron repulsion puts the energy above
    #    2 E_sigma (no repulsion), and correlation puts it below the mean field.
    e_mf = grid_mean_field(p["R"], p["Z1"], p["Z2"], p["soft"])[0]
    e_ex = grid_exact_two_electron(p["R"], p["Z1"], p["Z2"], p["soft"], n=36, box=14.0, verbose=False)[0]
    check("exact two-electron energy is below mean field (correlation)", 2 * e_grid < e_ex < e_mf,
          f"2 E_sigma {2 * e_grid:.5f} < exact {e_ex:.5f} < mean field {e_mf:.5f} Ha")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


# ============================================================================
# Command line
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Neural-network wavefunction (VMC) solver for the MO simulator's molecule.")
    ap.add_argument("--electrons", type=int, choices=(1, 2), default=2)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--walkers", type=int, default=WALKERS)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nn_results")
    ap.add_argument("--no-exact", action="store_true", help="skip the exact 4D grid reference")
    ap.add_argument("--curve", default=None, help="comma-separated bond lengths, e.g. 1,1.5,2,3,4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="single precision (faster; use on GPUs)")
    a = ap.parse_args(argv)
    print(f"JAX device: {jax.devices()[0].platform.upper()}, {'32' if FP32 else '64'}-bit")
    if a.selftest:
        return 0 if run_selftest() else 1
    iters = a.iterations or ITERATIONS[a.electrons]
    if a.curve:
        Rs = [float(v) for v in a.curve.split(",")]
        bond_curve(Rs, a.Z1, a.Z2, a.soft, iters, a.walkers, a.hidden, a.seed, a.out, exact=not a.no_exact)
    else:
        solve(a.R, a.Z1, a.Z2, a.soft, a.electrons, iters, a.walkers, a.hidden, a.seed,
              exact=not a.no_exact, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
