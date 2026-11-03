"""
mo_physics_jax.py - the MO Electron Simulator physics, running on JAX
---------------------------------------------------------------------
Same classes and behaviour as mo_physics.py (each class here extends the
NumPy one), but the expensive parts are compiled by JAX/XLA:

  * Simulation.advance(n): a whole animation frame (n time steps) runs as
    one compiled loop - waves (split-operator FFT), particles (velocity
    Verlet), electron-electron coupling and moving nuclei - without going
    back to Python between steps.
  * wave steps, imaginary-time relaxation, energies, expectation values and
    the Hartree potential are compiled functions.

The same code runs on CPU, or on a GPU if a CUDA build of JAX is installed
(pip install "jax[cuda12]"); JAX picks the device automatically.

Each new combination of settings (1 or 2 electrons, particle/wave, nuclei
moving or not, ...) is compiled the first time it is used, which takes a
second or two; after that it is fast.

Precision: 64-bit (like NumPy), so the physics checks pass unchanged:
    python mo_physics_jax.py          (runs mo_physics.run_selftest on this module)
"""

import os
import sys

import numpy as np

import jax

# 64-bit by default (same accuracy as NumPy). Set MO_JAX_FP32=1 (the windows'
# --fp32 option does this) for single precision: ~2x faster FFTs on a CPU, and
# the practical choice on consumer GPUs, whose 64-bit maths is much slower.
SINGLE_PRECISION = os.environ.get("MO_JAX_FP32", "") not in ("", "0")
jax.config.update("jax_enable_x64", not SINGLE_PRECISION)
import jax.numpy as jnp                                  # noqa: E402
from jax import lax                                      # noqa: E402

import mo_physics as base                                # noqa: E402
from mo_physics import (GRID_N, BOX_SIZE, DT, IMAG_DT, CLASSICAL_SUBSTEPS,  # noqa: E402,F401
                        ABSORBER_WIDTH, RELAX_TOL, RELAX_MAX_STEPS, ORBITAL_REGION_FRACTION,
                        DEFAULTS, START, PROTON_MASS, AU_TIME_FS, HARTREE_EV,
                        ClassicalElectron, Electron, orbital_region)

# Re-exported so this module can stand in for mo_physics.
__all__ = ["GRID_N", "BOX_SIZE", "DT", "IMAG_DT", "CLASSICAL_SUBSTEPS", "ABSORBER_WIDTH", "RELAX_TOL",
           "RELAX_MAX_STEPS", "ORBITAL_REGION_FRACTION", "DEFAULTS", "START", "PROTON_MASS",
           "AU_TIME_FS", "HARTREE_EV", "ClassicalElectron", "Electron", "orbital_region",
           "TwoCenterSystem", "QuantumElectron", "Simulation", "run_selftest", "device_name"]


CDTYPE = jnp.complex64 if SINGLE_PRECISION else jnp.complex128
FDTYPE = jnp.float32 if SINGLE_PRECISION else jnp.float64


def device_name():
    d = jax.devices()[0]
    name = f"{d.platform.upper()} ({d.device_kind})" if d.platform != "cpu" else "CPU"
    return name + (", 32-bit" if SINGLE_PRECISION else ", 64-bit")


# ============================================================================
# Pure functions (compiled by JAX)
# ============================================================================
def _grid_terms(X, Y2a, xn, Z):
    """Nuclear potential V on the grid and, per nucleus, the force per unit
    electron density G (x component)."""
    V = jnp.zeros_like(X)
    G = []
    for k in range(2):
        d = X - xn[k]
        inv = lax.rsqrt(d * d + Y2a)
        V = V - Z[k] * inv
        G.append(Z[k] * d * inv ** 3)
    return V, jnp.stack(G)


_grid_terms_jit = jax.jit(_grid_terms)


def _hartree(rho, kernel_k):
    return jnp.fft.irfft2(jnp.fft.rfft2(rho) * kernel_k, s=rho.shape)


_hartree_jit = jax.jit(_hartree)


def _rt_step(psi, half, kin, mask, absorb):
    psi = half * psi
    psi = jnp.fft.ifft2(kin * jnp.fft.fft2(psi))
    psi = half * psi
    return psi * mask if absorb else psi


@jax.jit
def _normalized(psi, dA):
    return psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dA)


def _rt_steps(psi, V, kin, mask, n, absorb):
    half = jnp.exp(-0.5j * DT * V)
    return lax.fori_loop(0, n, lambda _, p: _rt_step(p, half, kin, mask, absorb), psi)


_rt_steps_jit = jax.jit(_rt_steps, static_argnames=("absorb",))


def _imag_body(p, half, kin_im, proj, dA):
    p = half * p
    p = jnp.fft.ifft2(kin_im * jnp.fft.fft2(p))
    p = half * p
    for k in range(proj.shape[0]):                     # Gram-Schmidt against kept-out orbitals
        p = p - proj[k] * (jnp.vdot(proj[k], p) * dA)
    return p / jnp.sqrt(jnp.sum(jnp.abs(p) ** 2) * dA)


@jax.jit
def _imag_steps(psi, V, kin_im, proj, n, dA):
    half = jnp.exp(-0.5 * IMAG_DT * V)
    return lax.fori_loop(0, n, lambda _, p: _imag_body(p, half, kin_im, proj, dA), psi)


@jax.jit
def _relax_shared(phi, V, kin_im, kernel_k, n, dA):
    def body(_, p):
        half = jnp.exp(-0.5 * IMAG_DT * (V + _hartree(jnp.abs(p) ** 2, kernel_k)))
        return _imag_body(p, half, kin_im, jnp.zeros((0,) + p.shape, p.dtype), dA)
    return lax.fori_loop(0, n, body, phi)


@jax.jit
def _energy(psi, V, T):
    rho = jnp.abs(psi) ** 2
    w = jnp.abs(jnp.fft.fft2(psi)) ** 2
    return jnp.sum(w * T) / jnp.sum(w) + jnp.sum(rho * V) / jnp.sum(rho)


@jax.jit
def _expectations(psi, V, T, X, Y, KX, KY, dA):
    rho = jnp.abs(psi) ** 2
    total = jnp.sum(rho)
    w = jnp.abs(jnp.fft.fft2(psi)) ** 2
    wsum = jnp.sum(w)
    return jnp.stack([total * dA, jnp.sum(rho * X) / total, jnp.sum(rho * Y) / total,
                      jnp.sum(w * KX) / wsum, jnp.sum(w * KY) / wsum,
                      jnp.sum(w * T) / wsum, jnp.sum(rho * V) / total])


@jax.jit
def _density_pull(rho, G, dA):
    return jnp.sum(rho[None] * G, axis=(1, 2)) * dA


# -- pieces of one time step (used inside the compiled frame loop) -----------
def _nuclear_force_on_point(x, y, xn, Z, a2):
    fx = fy = 0.0
    for k in range(2):
        d = x - xn[k]
        s3 = (d * d + y * y + a2) ** 1.5
        fx = fx - Z[k] * d / s3
        fy = fy - Z[k] * y / s3
    return fx, fy


def _point_pull_on_nuclei(x, y, xn, Z, a2):
    return jnp.stack([Z[k] * (x - xn[k]) / ((x - xn[k]) ** 2 + y * y + a2) ** 1.5 for k in range(2)])


def _nn_forces(xn, Z):
    f = Z[0] * Z[1] / jnp.maximum(xn[1] - xn[0], 1e-3) ** 2
    return jnp.stack([-f, f])


def _interp(arr, x, y, dx, n):
    fx = x / dx + n / 2 - 0.5
    fy = y / dx + n / 2 - 0.5
    i, j = jnp.floor(fx), jnp.floor(fy)
    valid = (i >= 0) & (i < n - 1) & (j >= 0) & (j < n - 1)
    ic = jnp.clip(i, 0, n - 2).astype(jnp.int32)
    jc = jnp.clip(j, 0, n - 2).astype(jnp.int32)
    tx, ty = fx - i, fy - j
    val = (arr[jc, ic] * (1 - tx) * (1 - ty) + arr[jc, ic + 1] * tx * (1 - ty)
           + arr[jc + 1, ic] * (1 - tx) * ty + arr[jc + 1, ic + 1] * tx * ty)
    return jnp.where(valid, val, 0.0)


def _cloud_add(cloud, x, y, dx, n):
    i = jnp.floor(x / dx + n / 2)
    j = jnp.floor(y / dx + n / 2)
    valid = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    ic = jnp.clip(i, 0, n - 1).astype(jnp.int32)
    jc = jnp.clip(j, 0, n - 1).astype(jnp.int32)
    return cloud.at[jc, ic].add(jnp.where(valid, 1.0, 0.0))


def _verlet(pm, force_fn, clouds, idx, c, cloud_on):
    """Velocity Verlet over one DT for the particles pm (k, 4) = x, y, px, py.
    force_fn(pm) -> (k, 2). idx: which cloud each particle adds to."""
    h = DT / CLASSICAL_SUBSTEPS
    n = clouds.shape[-1]

    def body(_, carry):
        pm, f, clouds = carry
        p = pm[:, 2:] + 0.5 * h * f
        xy = pm[:, :2] + h * p
        f = force_fn(jnp.concatenate([xy, p], axis=1))
        p = p + 0.5 * h * f
        if cloud_on:
            for k, ci in enumerate(idx):
                clouds = clouds.at[ci].set(_cloud_add(clouds[ci], xy[k, 0], xy[k, 1], c["dx"], n))
        return jnp.concatenate([xy, p], axis=1), f, clouds

    pm, _, clouds = lax.fori_loop(0, CLASSICAL_SUBSTEPS, body, (pm, force_fn(pm), clouds))
    return pm, clouds


def _frame(state, c, n, cfg):
    """n time steps. cfg = (count, modes, move, absorb, cloud_on) is static."""
    count, modes, move, absorb, cloud_on = cfg
    active = list(range(count))
    a2 = c["a2"]
    ngrid = c["X"].shape[0]

    def nuclear_forces(st, G):
        f = _nn_forces(st["xn"], c["Z"])
        for k in active:
            if modes[k] == "p":
                f = f + _point_pull_on_nuclei(st["cl"][k, 0], st["cl"][k, 1], st["xn"], c["Z"], a2)
            else:
                f = f + _density_pull(jnp.abs(st["psi"][k]) ** 2, G, c["dA"])
        return f

    def field_of(st, k):
        if modes[k] == "p":
            x, y = st["cl"][k, 0], st["cl"][k, 1]
            return lax.rsqrt((c["X"] - x) ** 2 + (c["Y"] - y) ** 2 + a2)
        return _hartree(jnp.abs(st["psi"][k]) ** 2, c["kernel_k"])

    def step(_, st):
        st = dict(st)
        xn = st["xn"]
        if move:
            pn = st["pn"] + 0.5 * DT * st["fn"]
            xn = xn + DT * pn / c["mass"]
            st["xn"] = xn
        V, G = _grid_terms(c["X"], c["Y2a"], xn, c["Z"])
        psi, cl, clouds = st["psi"], st["cl"], st["clouds"]

        def nuc_force(pm_xy, k):
            return jnp.stack(_nuclear_force_on_point(pm_xy[0], pm_xy[1], xn, c["Z"], a2))

        if count == 1 and modes[0] == "p":
            pm, clouds = _verlet(cl[:1], lambda pm: nuc_force(pm[0], 0)[None], clouds, [0], c, cloud_on)
            cl = cl.at[:1].set(pm)
        elif count == 1:
            psi = psi.at[0].set(_rt_step(psi[0], jnp.exp(-0.5j * DT * V), c["kin"], c["mask"], absorb))
        elif modes[0] == modes[1] == "p":
            def pair(pm):
                d = pm[0, :2] - pm[1, :2]
                s3 = (jnp.sum(d * d) + a2) ** 1.5
                fee = d / s3
                return jnp.stack([nuc_force(pm[0], 0) + fee, nuc_force(pm[1], 1) - fee])
            cl, clouds = _verlet(cl, pair, clouds, [0, 1], c, cloud_on)
        else:
            fields = [field_of(st, k) for k in (0, 1)]
            for k in (0, 1):
                v_other = fields[1 - k]
                if modes[k] == "w":
                    half = jnp.exp(-0.5j * DT * (V + v_other))
                    psi = psi.at[k].set(_rt_step(psi[k], half, c["kin"], c["mask"], absorb))
                else:
                    gy, gx = jnp.gradient(v_other, c["dx"])

                    def force(pm, gx=gx, gy=gy, k=k):
                        x, y = pm[0, 0], pm[0, 1]
                        extra = jnp.stack([-_interp(gx, x, y, c["dx"], ngrid), -_interp(gy, x, y, c["dx"], ngrid)])
                        return (nuc_force(pm[0], k) + extra)[None]
                    pm, clouds = _verlet(cl[k:k + 1], force, clouds, [k], c, cloud_on)
                    cl = cl.at[k:k + 1].set(pm)
        st.update(psi=psi, cl=cl, clouds=clouds)
        if move:
            st["fn"] = nuclear_forces(st, G)
            st["pn"] = pn + 0.5 * DT * st["fn"]
        return st

    if move:
        _, G0 = _grid_terms(c["X"], c["Y2a"], state["xn"], c["Z"])
        state = dict(state, fn=nuclear_forces(state, G0))
    else:
        state = dict(state, fn=jnp.zeros(2))
    return lax.fori_loop(0, n, step, state)


_frame_jit = jax.jit(_frame, static_argnames=("cfg",))


# ============================================================================
# Classes (same interface as mo_physics)
# ============================================================================
class TwoCenterSystem(base.TwoCenterSystem):
    def __init__(self, R, Z1, Z2, soft, n=GRID_N, box=BOX_SIZE):
        super().__init__(R, Z1, Z2, soft, n, box)
        self.jX, self.jY = jnp.asarray(self.X), jnp.asarray(self.Y)
        self.jT = jnp.asarray(self.T)
        self.jKX, self.jKY = jnp.asarray(self.KX), jnp.asarray(self.KY)

    def set_params(self, R, Z1, Z2, soft):
        super().set_params(R, Z1, Z2, soft)
        self.j_ee_kernel = jnp.asarray(self.ee_kernel_k)
        self.jY2a = jnp.asarray(self._Y2a)

    def _grid_arrays(self):
        if getattr(self, "_grid_version", None) != self.version:
            if not hasattr(self, "jX"):                 # during the parent __init__
                self.jX = jnp.asarray(self.X)
            jV, jG = _grid_terms_jit(self.jX, self.jY2a, jnp.array([self.x1, self.x2]),
                                     jnp.array([self.Z1, self.Z2]))
            self._jV, self._jG, self._V, self._G = jV, jG, None, None
            self._grid_version = self.version

    @property
    def jV(self):
        self._grid_arrays()
        return self._jV

    @property
    def V(self):                                       # NumPy copy for drawing / NumPy code
        self._grid_arrays()
        if self._V is None:
            self._V = np.asarray(self._jV)
        return self._V

    @property
    def G(self):
        self._grid_arrays()
        if self._G is None:
            self._G = list(np.asarray(self._jG))
        return self._G

    def hartree(self, rho):
        return _hartree_jit(jnp.asarray(rho), self.j_ee_kernel)

    def nuclear_forces_density(self, rho):
        self._grid_arrays()
        return [float(v) for v in _density_pull(jnp.asarray(rho), self._jG, self.dA)]


class QuantumElectron(base.QuantumElectron):
    def __init__(self, system, absorb=True):
        super().__init__(system, absorb)
        self.jmask = jnp.asarray(self._mask)

    def rebuild(self):
        s = self.sys
        self.kin = jnp.exp(-1j * DT * s.jT)
        self.kin_im = jnp.exp(-IMAG_DT * s.jT)

    def normalized(self, psi):
        return _normalized(jnp.asarray(psi, dtype=CDTYPE), self.sys.dA)

    def _v(self, v_extra):
        return self.sys.jV if v_extra is None else self.sys.jV + jnp.asarray(v_extra)

    def step(self, n=1, v_extra=None):
        self.psi = _rt_steps_jit(jnp.asarray(self.psi), self._v(v_extra), self.kin, self.jmask,
                                 n, absorb=bool(self.absorb))

    def imag_steps(self, psi, n, project_out=(), v_extra=None):
        psi = jnp.asarray(psi, dtype=CDTYPE)
        proj = (jnp.stack([jnp.asarray(p) for p in project_out]) if len(project_out)
                else jnp.zeros((0,) + psi.shape, psi.dtype))
        return _imag_steps(psi, self._v(v_extra), self.kin_im, proj, n, self.sys.dA)

    def energy(self, psi, v_extra=None):
        return float(_energy(jnp.asarray(psi), self._v(v_extra), self.sys.jT))

    def expectations(self):
        s = self.sys
        v = np.asarray(_expectations(jnp.asarray(self.psi), s.jV, s.jT, s.jX, s.jY, s.jKX, s.jKY, s.dA))
        return dict(norm=v[0], x=v[1], y=v[2], px=v[3], py=v[4], T=v[5], V=v[6], E=v[5] + v[6])

    def measure(self, rng):
        saved = self.psi
        self.psi = np.asarray(saved)
        try:
            return super().measure(rng)
        finally:
            self.psi = saved


class Simulation(base.Simulation):
    wave_class = QuantumElectron

    def __init__(self, system, absorb=True):
        super().__init__(system, absorb)
        self._dummy_psi = jnp.zeros((system.n, system.n), CDTYPE)

    def step(self, clouds=(None, None)):
        self.advance(1, clouds)

    def advance(self, n, clouds=(None, None)):
        """n time steps as one compiled loop."""
        s = self.sys
        els = self.electrons
        modes = tuple("w" if (k < self.count and e.mode == "wave") else "p" for k, e in enumerate(els))
        cloud_on = clouds[0] is not None
        cfg = (self.count, modes, bool(self.nuclei_move), bool(els[0].wave.absorb), cloud_on)
        if "w" in modes:
            psi = jnp.stack([jnp.asarray(e.wave.psi) if m == "w" else self._dummy_psi
                             for e, m in zip(els, modes)])
        else:                                          # no waves: tiny placeholder
            psi = jnp.zeros((2, 1, 1), CDTYPE)
        cl = jnp.array([[e.particle.x, e.particle.y, e.particle.px, e.particle.py]
                        if (m == "p" and e.particle is not None) else [0.0, 0.0, 0.0, 0.0]
                        for e, m in zip(els, modes)])
        cl_clouds = jnp.asarray(np.stack(clouds)) if cloud_on else jnp.zeros((2, 1, 1))
        state = dict(psi=psi, cl=cl, clouds=cl_clouds,
                     xn=jnp.array([s.x1, s.x2]), pn=jnp.array(self.nuc_p, dtype=FDTYPE))
        consts = dict(X=s.jX, Y=s.jY, Y2a=s.jY2a, kin=els[0].wave.kin, mask=els[0].wave.jmask,
                      kernel_k=s.j_ee_kernel, Z=jnp.array([s.Z1, s.Z2]), a2=s.soft ** 2,
                      mass=float(self.nuclear_mass), dx=s.dx, dA=s.dA)
        out = _frame_jit(state, consts, n, cfg=cfg)

        cl = np.asarray(out["cl"])
        for k, (e, m) in enumerate(zip(els, modes)):
            if k >= self.count:
                continue
            if m == "w":
                e.wave.psi = out["psi"][k]
            else:
                c = e.particle
                c.x, c.y, c.px, c.py = (float(v) for v in cl[k])
        if cloud_on:
            res = np.asarray(out["clouds"])
            for k in range(2):
                clouds[k][...] = res[k]
        if self.nuclei_move:
            xn = np.asarray(out["xn"])
            s.set_positions(float(xn[0]), float(xn[1]))
            self.nuc_p = [float(v) for v in np.asarray(out["pn"])]
        for e in els:
            if e.particle is not None:
                e.particle.refresh_force()
        self.t += n * DT

    def relax_shared(self, n):
        s = self.sys
        q = self.electrons[0].wave
        phi = _relax_shared(jnp.asarray(q.psi), s.jV, q.kin_im, s.j_ee_kernel, n, s.dA)
        q.psi = phi
        self.electrons[1].wave.psi = phi


def run_selftest(m=None):
    """The NumPy module's physics checks, run on this (JAX) module."""
    if SINGLE_PRECISION:
        print("The physics checks need 64-bit precision: run them without --fp32 / MO_JAX_FP32.")
        return False
    return base.run_selftest(m or sys.modules[__name__])


if __name__ == "__main__":
    print("JAX device:", device_name())
    sys.exit(0 if run_selftest() else 1)
