"""
mo_physics.py - physics core of the MO Electron Simulator (NumPy version)
-------------------------------------------------------------------------
No user interface in here; the windows (MO-Electron-Simulator.py,
MO-Electron-Simulator-Qt.py, ...) import this module, and mo_physics_jax.py
provides the same classes running on JAX.

  TwoCenterSystem    two nuclei on the x axis + the numerical grid
  ClassicalElectron  point electron (velocity Verlet)
  QuantumElectron    wavefunction on the grid (split-operator FFT)
  Electron           one electron that is either a particle or a wave
  Simulation         one or two electrons (+ optionally moving nuclei)
  orbital_region     the region holding 90 % of an orbital
  run_selftest       physics checks; run_selftest(module) checks any backend

Units: atomic units (hbar = m_e = e = 4*pi*eps0 = 1); see the main program
for the model assumptions.
"""

import math
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Numerical settings
# ----------------------------------------------------------------------------
GRID_N = 256              # grid points per side for each wavefunction
BOX_SIZE = 40.0           # box side length, bohr (grid spacing = BOX_SIZE / GRID_N)
DT = 0.02                 # time step, a.u. (particles and waves)
IMAG_DT = 0.05            # imaginary time step used by "Relax"
CLASSICAL_SUBSTEPS = 10   # Newton steps per DT (keeps energy conserved near the nuclei)
ABSORBER_WIDTH = 4.0      # width of the absorbing border, bohr
RELAX_TOL = 1e-8          # hartree; relaxing stops when the energy changes less than this
RELAX_MAX_STEPS = 6000    # give up after this many imaginary-time steps
ORBITAL_REGION_FRACTION = 0.9   # the "inside the orbital" region holds this much of sigma

DEFAULTS = dict(
    R=2.0,        # bond length, bohr
    Z1=1.0,       # charge of left nucleus
    Z2=1.0,       # charge of right nucleus
    soft=0.5,     # soft-core parameter a, bohr
    sigma=0.6,    # wave packet width, bohr
    steps=5,      # time steps per animation frame (simulation speed)
    rate=0.05,    # auto switch events per a.u. of time
    mass=1836,    # mass of each nucleus in electron masses (hydrogen: 1836)
)
START = [((0.0, 1.2), (0.75, 0.0)),      # electron 1: position (bohr), momentum (a.u.)
         ((0.0, -1.2), (-0.75, 0.0))]    # electron 2

PROTON_MASS = 1836.15     # electron masses
AU_TIME_FS = 0.02418884
HARTREE_EV = 27.211386


# ============================================================================
# Physics (no user interface code below until run_app)
# ============================================================================
class TwoCenterSystem:
    """Two fixed nuclei on the x axis at -R/2 and +R/2, plus the numerical grid."""

    def __init__(self, R, Z1, Z2, soft, n=GRID_N, box=BOX_SIZE):
        self.n = n
        self.box = box
        self.dx = box / n
        self.dA = self.dx * self.dx
        # Grid symmetric about 0 so left/right parity is exact on the grid.
        self.x = (np.arange(n) - n / 2 + 0.5) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.x)          # arrays indexed [y, x]
        k = 2 * np.pi * np.fft.fftfreq(n, self.dx)
        self.KX, self.KY = np.meshgrid(k, k)
        self.T = 0.5 * (self.KX ** 2 + self.KY ** 2)           # kinetic energy in k space
        d = np.fft.fftfreq(n, 1.0 / (n * self.dx))             # displacements 0, dx, ..., -dx
        self._DX, self._DY = np.meshgrid(d, d)
        self.version = 0          # changes whenever the potential changes (also when nuclei move)
        self.params_version = 0   # changes only when charges / softening / R are set
        self.set_params(R, Z1, Z2, soft)

    def set_params(self, R, Z1, Z2, soft):
        """Set charges and softening, and place the nuclei at -R/2 and +R/2."""
        self.Z1, self.Z2, self.soft = float(Z1), float(Z2), float(soft)
        self._Y2a = self.Y ** 2 + self.soft ** 2
        # Electron-electron kernel 1/sqrt(d^2 + a^2) in k space, for the
        # Hartree potential of a charge cloud (exact for |r - r'| < box / 2).
        kernel = 1.0 / np.sqrt(self._DX ** 2 + self._DY ** 2 + self.soft ** 2)
        self.ee_kernel_k = np.fft.rfft2(kernel) * self.dA
        self.params_version += 1
        self.set_positions(-float(R) / 2, float(R) / 2)

    def set_positions(self, x1, x2):
        """Move the nuclei along the x axis (x1 < x2) and update the potential."""
        self.x1, self.x2 = float(x1), float(x2)
        self.R = self.x2 - self.x1
        self.nuclei = [(self.x1, self.Z1), (self.x2, self.Z2)]
        self.version += 1

    def _grid_arrays(self):
        # Grid potential V and nuclear-force kernels G, rebuilt only when asked for
        # (moving nuclei with particle electrons never need them).
        if getattr(self, "_grid_version", None) != self.version:
            V = 0.0
            G = []         # force on each nucleus per unit electron density, x component
            for xn, z in self.nuclei:
                d = self.X - xn
                inv = 1.0 / np.sqrt(d * d + self._Y2a)
                V = V - z * inv
                G.append(z * d * inv ** 3)
            self._V, self._G, self._grid_version = V, G, self.version

    @property
    def V(self):
        self._grid_arrays()
        return self._V

    @property
    def G(self):
        self._grid_arrays()
        return self._G

    # -- electron-nucleus ---------------------------------------------------
    def potential(self, x, y):
        a2 = self.soft ** 2
        v = 0.0
        for xn, z in self.nuclei:
            v = v - z / np.sqrt((x - xn) ** 2 + y ** 2 + a2)
        return v

    def force(self, x, y):
        """Coulomb force of the nuclei on an electron, F = -grad V (plain floats, fast)."""
        a2 = self.soft ** 2
        fx = fy = 0.0
        for xn, z in self.nuclei:
            dx = x - xn
            s3 = (dx * dx + y * y + a2) ** 1.5
            fx -= z * dx / s3
            fy -= z * y / s3
        return fx, fy

    def nuclear_repulsion(self):
        return self.Z1 * self.Z2 / max(self.R, 1e-3)

    # -- forces on the nuclei (along the bond axis) -------------------------
    def nn_forces(self):
        """Nucleus-nucleus Coulomb repulsion: pushes nucleus 1 left, nucleus 2 right."""
        f = self.Z1 * self.Z2 / max(self.R, 1e-3) ** 2
        return [-f, f]

    def nuclear_forces_point(self, x, y):
        """Pull of a point electron at (x, y) on each nucleus."""
        a2 = self.soft ** 2
        return [z * (x - xn) / ((x - xn) ** 2 + y * y + a2) ** 1.5 for xn, z in self.nuclei]

    def nuclear_forces_density(self, rho):
        """Pull of an electron charge cloud rho on each nucleus."""
        return [float(np.sum(rho * g) * self.dA) for g in self.G]

    # -- electron-electron --------------------------------------------------
    def ee_energy(self, x1, y1, x2, y2):
        return 1.0 / math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + self.soft ** 2)

    def ee_force(self, x1, y1, x2, y2):
        """Repulsive force on electron 1 from electron 2."""
        dx, dy = x1 - x2, y1 - y2
        s3 = (dx * dx + dy * dy + self.soft ** 2) ** 1.5
        return dx / s3, dy / s3

    def ee_point(self, x, y):
        """Grid potential felt by an electron due to a point electron at (x, y)."""
        return 1.0 / np.sqrt((self.X - x) ** 2 + (self.Y - y) ** 2 + self.soft ** 2)

    def hartree(self, rho):
        """Grid potential felt by an electron due to the charge cloud rho."""
        return np.fft.irfft2(np.fft.rfft2(rho) * self.ee_kernel_k, s=rho.shape)

    # -- grid helpers -------------------------------------------------------
    def grid_index(self, x, y):
        """(row, col) of the grid cell containing (x, y), or None if outside."""
        i = int(math.floor(x / self.dx + self.n / 2))
        j = int(math.floor(y / self.dx + self.n / 2))
        if 0 <= i < self.n and 0 <= j < self.n:
            return j, i
        return None

    def interp(self, arr, x, y):
        """Bilinear interpolation of a grid array at (x, y); 0 outside the grid."""
        fx = x / self.dx + self.n / 2 - 0.5
        fy = y / self.dx + self.n / 2 - 0.5
        i, j = int(math.floor(fx)), int(math.floor(fy))
        if not (0 <= i < self.n - 1 and 0 <= j < self.n - 1):
            return 0.0
        tx, ty = fx - i, fy - j
        return float(arr[j, i] * (1 - tx) * (1 - ty) + arr[j, i + 1] * tx * (1 - ty)
                     + arr[j + 1, i] * (1 - tx) * ty + arr[j + 1, i + 1] * tx * ty)


class ClassicalElectron:
    """Point electron, integrated with velocity Verlet."""

    def __init__(self, system, pos, mom):
        self.sys = system
        self.x, self.y = float(pos[0]), float(pos[1])
        self.px, self.py = float(mom[0]), float(mom[1])
        self.refresh_force()

    def refresh_force(self):
        self.fx, self.fy = self.sys.force(self.x, self.y)

    def step(self, dt, substeps=CLASSICAL_SUBSTEPS, cloud=None, extra_force=None):
        """Advance by dt. extra_force(x, y) -> (fx, fy) adds a force on top of the nuclei's."""
        nuclear = self.sys.force
        if extra_force is None:
            force = nuclear
        else:
            def force(x, y):
                fx, fy = nuclear(x, y)
                ex, ey = extra_force(x, y)
                return fx + ex, fy + ey
        h = dt / substeps
        self.fx, self.fy = force(self.x, self.y)
        for _ in range(substeps):
            self.px += 0.5 * h * self.fx
            self.py += 0.5 * h * self.fy
            self.x += h * self.px
            self.y += h * self.py
            self.fx, self.fy = force(self.x, self.y)
            self.px += 0.5 * h * self.fx
            self.py += 0.5 * h * self.fy
            if cloud is not None:
                idx = self.sys.grid_index(self.x, self.y)
                if idx is not None:
                    cloud[idx] += 1.0

    def energies(self):
        """Kinetic, nuclear potential and their sum (no electron-electron part)."""
        t = 0.5 * (self.px ** 2 + self.py ** 2)
        v = float(self.sys.potential(self.x, self.y))
        return t, v, t + v


def step_particle_pair(a, b, dt, substeps=CLASSICAL_SUBSTEPS, clouds=(None, None)):
    """Velocity Verlet for two point electrons that also repel each other."""
    s = a.sys
    h = dt / substeps

    def forces():
        fax, fay = s.force(a.x, a.y)
        fbx, fby = s.force(b.x, b.y)
        ex, ey = s.ee_force(a.x, a.y, b.x, b.y)
        return fax + ex, fay + ey, fbx - ex, fby - ey

    a.fx, a.fy, b.fx, b.fy = forces()
    for _ in range(substeps):
        for c in (a, b):
            c.px += 0.5 * h * c.fx
            c.py += 0.5 * h * c.fy
            c.x += h * c.px
            c.y += h * c.py
        a.fx, a.fy, b.fx, b.fy = forces()
        for c, cloud in zip((a, b), clouds):
            c.px += 0.5 * h * c.fx
            c.py += 0.5 * h * c.fy
            if cloud is not None:
                idx = s.grid_index(c.x, c.y)
                if idx is not None:
                    cloud[idx] += 1.0


class QuantumElectron:
    """Wavefunction on the grid, propagated with the split-operator FFT method.
    v_extra (optional grid array) is added to the nuclear potential, e.g. the
    repulsion from the other electron."""

    def __init__(self, system, absorb=True):
        self.sys = system
        self.absorb = absorb
        self.psi = None
        self._mask = self._make_mask()
        self.rebuild()

    def _make_mask(self):
        s = self.sys
        edge = s.box / 2 - np.abs(s.x)            # distance to the nearest wall
        m = np.ones_like(s.x)
        inside = edge < ABSORBER_WIDTH
        m[inside] = np.cos(0.5 * np.pi * (ABSORBER_WIDTH - edge[inside]) / ABSORBER_WIDTH) ** 0.125
        return np.outer(m, m)

    def rebuild(self):
        """Call after the molecule changes. The potential factors are rebuilt
        automatically (lazily) whenever the potential changes, e.g. nuclei moving."""
        s = self.sys
        self.kin = np.exp(-1j * DT * s.T)
        self.kin_im = np.exp(-IMAG_DT * s.T)
        self._hv = self._hvi = (None, None)

    @property
    def half_v(self):
        if self._hv[0] != self.sys.version:
            self._hv = (self.sys.version, np.exp(-0.5j * DT * self.sys.V))
        return self._hv[1]

    @property
    def half_v_im(self):
        if self._hvi[0] != self.sys.version:
            self._hvi = (self.sys.version, np.exp(-0.5 * IMAG_DT * self.sys.V))
        return self._hvi[1]

    # -- creating states ----------------------------------------------------
    def normalized(self, psi):
        return psi / math.sqrt(float(np.sum(np.abs(psi) ** 2)) * self.sys.dA)

    def gaussian(self, pos, mom, sigma):
        s = self.sys
        x0, y0 = pos
        px, py = mom
        r2 = (s.X - x0) ** 2 + (s.Y - y0) ** 2
        psi = np.exp(-r2 / (4 * sigma ** 2) + 1j * (px * s.X + py * s.Y))
        return self.normalized(psi)

    def from_particle(self, pos, mom, sigma):
        self.psi = self.gaussian(pos, mom, sigma)

    def lcao(self, sign):
        """Crude LCAO guess: 1s-like orbital on each nucleus, added (+1) or subtracted (-1)."""
        s = self.sys
        a, b = [np.exp(-z * np.sqrt((s.X - xn) ** 2 + s.Y ** 2)) for xn, z in s.nuclei]
        return self.normalized((a + sign * b).astype(complex))

    # -- time evolution -----------------------------------------------------
    def step(self, n=1, v_extra=None):
        psi = self.psi
        half_v = self.half_v if v_extra is None else np.exp(-0.5j * DT * (self.sys.V + v_extra))
        fft2, ifft2 = np.fft.fft2, np.fft.ifft2
        for _ in range(n):
            psi = half_v * psi
            psi = ifft2(self.kin * fft2(psi))
            psi = half_v * psi
            if self.absorb:
                psi = psi * self._mask
        self.psi = psi

    def imag_steps(self, psi, n, project_out=(), v_extra=None):
        """Imaginary-time steps: every component decays as exp(-E t), so the
        lowest state (not projected out) survives. Returns the new array."""
        dA = self.sys.dA
        half_v = self.half_v_im if v_extra is None else np.exp(-0.5 * IMAG_DT * (self.sys.V + v_extra))
        fft2, ifft2 = np.fft.fft2, np.fft.ifft2
        for _ in range(n):
            psi = half_v * psi
            psi = ifft2(self.kin_im * fft2(psi))
            psi = half_v * psi
            for phi in project_out:
                psi = psi - phi * (np.vdot(phi, psi) * dA)
            psi = self.normalized(psi)
        return psi

    def energy(self, psi, v_extra=None):
        s = self.sys
        rho = np.abs(psi) ** 2
        w = np.abs(np.fft.fft2(psi)) ** 2
        v = s.V if v_extra is None else s.V + v_extra
        return float(np.sum(w * s.T) / np.sum(w) + np.sum(rho * v) / np.sum(rho))

    def solve_state(self, seed, project_out=(), v_extra=None, tol=RELAX_TOL,
                    max_steps=8000, chunk=25):
        """Relax `seed` until its energy stops changing. Returns (psi, E)."""
        psi = self.normalized(seed)
        for phi in project_out:
            psi = psi - phi * (np.vdot(phi, psi) * self.sys.dA)
        e_old = np.inf
        e = self.energy(psi, v_extra)
        done = 0
        while abs(e - e_old) > tol and done < max_steps:
            psi = self.imag_steps(psi, chunk, project_out, v_extra)
            done += chunk
            e_old, e = e, self.energy(psi, v_extra)
        return psi, e

    # -- measuring ----------------------------------------------------------
    def expectations(self):
        s = self.sys
        rho = np.abs(self.psi) ** 2
        total = float(np.sum(rho))
        w = np.abs(np.fft.fft2(self.psi)) ** 2
        wsum = float(np.sum(w))
        t = float(np.sum(w * s.T) / wsum)
        v = float(np.sum(rho * s.V) / total)
        return dict(
            norm=total * s.dA,
            x=float(np.sum(rho * s.X) / total),
            y=float(np.sum(rho * s.Y) / total),
            px=float(np.sum(w * s.KX) / wsum),
            py=float(np.sum(w * s.KY) / wsum),
            T=t, V=v, E=t + v,
        )

    def measure(self, rng):
        """Collapse: draw a position from |psi(r)|^2 and a momentum from |psi(k)|^2."""
        s = self.sys
        rho = (np.abs(self.psi) ** 2).ravel()
        idx = rng.choice(rho.size, p=rho / rho.sum())
        j, i = divmod(idx, s.n)
        x = s.x[i] + rng.uniform(-0.5, 0.5) * s.dx
        y = s.x[j] + rng.uniform(-0.5, 0.5) * s.dx
        w = (np.abs(np.fft.fft2(self.psi)) ** 2).ravel()
        idx = rng.choice(w.size, p=w / w.sum())
        dk = 2 * np.pi / s.box
        px = s.KX.flat[idx] + rng.uniform(-0.5, 0.5) * dk
        py = s.KY.flat[idx] + rng.uniform(-0.5, 0.5) * dk
        return (float(x), float(y)), (float(px), float(py))


class Electron:
    """One electron that is currently either a particle or a wave."""

    def __init__(self, system, absorb=True, wave_class=None):
        self.sys = system
        self.mode = "particle"
        self.particle = None
        self.wave = (wave_class or QuantumElectron)(system, absorb)

    def set_particle(self, pos, mom):
        self.mode = "particle"
        self.particle = ClassicalElectron(self.sys, pos, mom)

    def set_wave(self, pos, mom, sigma):
        self.mode = "wave"
        self.wave.from_particle(pos, mom, sigma)

    def to_wave(self, sigma):
        c = self.particle
        self.set_wave((c.x, c.y), (c.px, c.py), sigma)

    def measure(self, rng):
        pos, mom = self.wave.measure(rng)
        self.set_particle(pos, mom)
        return pos, mom

    def density(self):
        return np.abs(self.wave.psi) ** 2


class Simulation:
    """One or two electrons around the two nuclei; each a particle or a wave."""
    wave_class = None             # QuantumElectron; the JAX backend substitutes its own

    def __init__(self, system, absorb=True):
        self.sys = system
        self.electrons = [Electron(system, absorb, self.wave_class) for _ in range(2)]
        self.count = 1
        self.t = 0.0
        self.nuclei_move = False          # True: nuclei vibrate along the bond axis
        self.nuclear_mass = PROTON_MASS   # mass of each nucleus, in electron masses
        self.nuc_p = [0.0, 0.0]           # nuclear momenta along x

    def active(self):
        return self.electrons[:self.count]

    def other(self, e):
        if self.count < 2:
            return None
        a, b = self.electrons
        return b if e is a else a

    def rebuild(self):
        """Call after the molecule changes."""
        for e in self.electrons:
            e.wave.rebuild()
            if e.particle is not None:
                e.particle.refresh_force()

    def set_absorb(self, flag):
        for e in self.electrons:
            e.wave.absorb = flag

    def field_of(self, e):
        """Grid potential that electron e creates for the other electron."""
        if e.mode == "particle":
            return self.sys.ee_point(e.particle.x, e.particle.y)
        return self.sys.hartree(e.density())

    def advance(self, n, clouds=(None, None)):
        """n time steps (the windows call this once per frame)."""
        for _ in range(n):
            self.step(clouds)

    def nuclear_forces(self):
        """Total force on each nucleus along the bond axis: the other nucleus
        pushes, the electrons pull (point charge or charge cloud)."""
        s = self.sys
        f = s.nn_forces()
        for e in self.active():
            if e.mode == "particle":
                g = s.nuclear_forces_point(e.particle.x, e.particle.y)
            else:
                g = s.nuclear_forces_density(e.density())
            f = [f[0] + g[0], f[1] + g[1]]
        return f

    def step(self, clouds=(None, None)):
        """Advance everything by DT. With moving nuclei: half kick, move the
        nuclei, evolve the electrons in the new potential, half kick."""
        if not self.nuclei_move:
            self._step_electrons(clouds)
            self.t += DT
            return
        s, m = self.sys, self.nuclear_mass
        f = self.nuclear_forces()
        p = [self.nuc_p[k] + 0.5 * DT * f[k] for k in range(2)]
        s.set_positions(s.x1 + DT * p[0] / m, s.x2 + DT * p[1] / m)
        self._step_electrons(clouds)
        f = self.nuclear_forces()
        self.nuc_p = [p[k] + 0.5 * DT * f[k] for k in range(2)]
        self.t += DT

    def _step_electrons(self, clouds):
        s = self.sys
        els = self.active()
        if len(els) == 1:
            e = els[0]
            if e.mode == "particle":
                e.particle.step(DT, cloud=clouds[0])
            else:
                e.wave.step(1)
        elif all(e.mode == "particle" for e in els):
            step_particle_pair(els[0].particle, els[1].particle, DT, clouds=clouds)
        else:
            # Fields from the state at the start of the step, then move each electron.
            fields = [self.field_of(e) for e in els]
            for k, e in enumerate(els):
                v_other = fields[1 - k]
                if e.mode == "wave":
                    e.wave.step(1, v_extra=v_other)
                else:
                    gy, gx = np.gradient(v_other, s.dx)

                    def extra(x, y, gx=gx, gy=gy):
                        return -s.interp(gx, x, y), -s.interp(gy, x, y)
                    e.particle.step(DT, cloud=clouds[k], extra_force=extra)

    def relax(self, idxs, n, project=None):
        """n imaginary-time steps for the electrons in idxs (must be waves); the
        others stay frozen. project = {index: [orbitals to keep out]}."""
        project = project or {}
        if self.count == 1:
            q = self.electrons[0].wave
            q.psi = q.imag_steps(q.psi, n, project.get(0, ()))
            return
        for _ in range(n):
            for i in idxs:
                e = self.electrons[i]
                v = self.field_of(self.other(e))
                e.wave.psi = e.wave.imag_steps(e.wave.psi, 1, project.get(i, ()), v)

    def relax_shared(self, n):
        """Both electrons in one shared orbital (opposite spins, closed shell):
        n imaginary-time steps of that orbital in the nuclei + its own Hartree field."""
        q = self.electrons[0].wave
        phi = q.psi
        for _ in range(n):
            phi = q.imag_steps(phi, 1, (), self.sys.hartree(np.abs(phi) ** 2))
        q.psi = phi
        self.electrons[1].wave.psi = phi.copy()

    def observe(self):
        """Per-electron positions/energies and the total energy."""
        s = self.sys
        infos = []
        t_sum = vn_sum = 0.0
        for e in self.active():
            if e.mode == "particle":
                c = e.particle
                t, v, _ = c.energies()
                info = dict(mode="particle", x=c.x, y=c.y, px=c.px, py=c.py, T=t, V=v, norm=1.0)
            else:
                info = e.wave.expectations()
                info["mode"] = "wave"
                info["rho"] = e.density()
            infos.append(info)
            t_sum += info["T"]
            vn_sum += info["V"]
        vee = 0.0
        if self.count == 2:
            a, b = infos
            if a["mode"] == b["mode"] == "particle":
                vee = s.ee_energy(a["x"], a["y"], b["x"], b["y"])
            elif a["mode"] == b["mode"] == "wave":
                vee = float(np.sum(a["rho"] * s.hartree(b["rho"])) * s.dA)
            else:
                w, p = (a, b) if a["mode"] == "wave" else (b, a)
                vee = float(np.sum(w["rho"] * s.ee_point(p["x"], p["y"])) * s.dA)
        out = dict(electrons=infos, T=t_sum, V=vn_sum + vee, Vee=vee, E=t_sum + vn_sum + vee,
                   E_el=t_sum + vn_sum + vee, R=s.R, Vnn=s.nuclear_repulsion(), Tnuc=0.0)
        if self.nuclei_move:
            # Everything moves, so the conserved energy includes the nuclei.
            tn = sum(p * p for p in self.nuc_p) / (2 * self.nuclear_mass)
            out.update(Tnuc=tn, T=t_sum + tn, V=out["V"] + out["Vnn"], E=out["E"] + out["Vnn"] + tn)
        return out


def orbital_region(psi, dA, frac=ORBITAL_REGION_FRACTION):
    """Mask of the smallest region holding `frac` of |psi|^2, and its density threshold."""
    rho = np.abs(psi) ** 2
    flat = np.sort(rho.ravel())[::-1]
    cum = np.cumsum(flat) * dA
    thr = flat[min(int(np.searchsorted(cum, frac * cum[-1])), flat.size - 1)]
    return rho >= thr, thr


# ============================================================================
# Headless physics checks:  python MO-Electron-Simulator.py --selftest
# ============================================================================


def run_selftest(m=None):
    """Physics checks. m = the physics module to test (default: this one)."""
    m = m or sys.modules[__name__]
    print(f"Testing {m.__name__}")
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    p = DEFAULTS
    s = m.TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
    (pos1, mom1), (pos2, mom2) = START

    print("-- one electron --")
    # 1. Particle: total energy conserved by velocity Verlet.
    c = m.ClassicalElectron(s, pos1, mom1)
    e0 = c.energies()[2]
    emax = 0.0
    for _ in range(20000):                      # 400 a.u. of time
        c.step(DT)
        emax = max(emax, abs(c.energies()[2] - e0))
    check("classical energy conservation", emax < 1e-3 * abs(e0),
          f"E0 = {e0:.6f} Ha, max drift = {emax:.2e} Ha over 400 a.u.")

    # 2. Wave: norm conserved without the absorber.
    q = m.QuantumElectron(s, absorb=False)
    q.from_particle(pos1, mom1, p["sigma"])
    q.step(500)
    n = q.expectations()["norm"]
    check("quantum norm conservation", abs(n - 1) < 1e-10, f"norm after 10 a.u. = {n:.14f}")

    # 3. Particle -> wave keeps position and momentum.
    q.from_particle((1.0, -0.5), (0.3, 0.8), 0.6)
    ex = q.expectations()
    err = max(abs(ex["x"] - 1.0), abs(ex["y"] + 0.5), abs(ex["px"] - 0.3), abs(ex["py"] - 0.8))
    check("become-wave keeps <r>, <p>", err < 1e-3,
          f"<r> = ({ex['x']:.4f}, {ex['y']:.4f}), <p> = ({ex['px']:.4f}, {ex['py']:.4f})")

    # 4. Orbitals: bonding is gerade, antibonding is ungerade, E_sigma < E_sigma*.
    flip = lambda a: a[:, ::-1]                 # x -> -x (grid is symmetric)
    g, eg = q.solve_state(q.lcao(+1))
    u, eu = q.solve_state(q.lcao(-1), project_out=[g])
    sym_g = np.max(np.abs(g - flip(g))) / np.max(np.abs(g))
    sym_u = np.max(np.abs(u + flip(u))) / np.max(np.abs(u))
    check("sigma (bonding) is symmetric", sym_g < 1e-6, f"E = {eg:.6f} Ha, asymmetry {sym_g:.1e}")
    check("sigma* (antibonding) is antisymmetric", sym_u < 1e-6,
          f"E = {eu:.6f} Ha, antisymmetry error {sym_u:.1e}")
    check("E(sigma) < E(sigma*)", eg < eu, f"gap = {eu - eg:.4f} Ha = {(eu - eg) * HARTREE_EV:.3f} eV")
    g2, eg2 = q.solve_state(q.gaussian((3.0, 2.0), (0.5, -0.2), 0.6))
    check("relax from a wave packet finds sigma", abs(eg2 - eg) < 1e-6, f"E = {eg2:.6f} Ha")

    # 5. Measuring the orbital gives particles with the same mean energy.
    q.psi = g
    rng = np.random.default_rng(1)
    energies, xs = [], []
    for _ in range(4000):
        (x, y), (px, py) = q.measure(rng)
        energies.append(0.5 * (px * px + py * py) + float(s.potential(x, y)))
        xs.append(x)
    em, xm = float(np.mean(energies)), float(np.mean(xs))
    check("measurement statistics", abs(em - eg) < 0.1 and abs(xm) < 0.1,
          f"mean particle energy = {em:.3f} Ha (orbital {eg:.3f}), mean x = {xm:.3f}")

    # 6. The "inside the orbital" region holds the requested share of sigma.
    mask, _ = m.orbital_region(g, s.dA)
    share = float(np.sum(np.abs(g) ** 2 * mask) * s.dA)
    check("orbital region", abs(share - ORBITAL_REGION_FRACTION) < 0.01, f"sigma inside = {share:.3f}")

    print("-- two electrons --")
    # 7. Hartree potential of a small charge cloud looks like a point charge from afar.
    q.from_particle((0.0, 0.0), (0.0, 0.0), 0.3)
    vh = s.hartree(np.abs(q.psi) ** 2)
    got, want = s.interp(vh, 5.0, 0.0), 1.0 / math.sqrt(25 + s.soft ** 2)
    check("Hartree potential", abs(got / want - 1) < 0.02, f"V_H(5, 0) = {got:.4f}, point charge {want:.4f}")

    # 8. Two particles: total energy (including their repulsion) conserved.
    sim = m.Simulation(s, absorb=False)
    sim.count = 2
    sim.electrons[0].set_particle(pos1, mom1)
    sim.electrons[1].set_particle(pos2, mom2)
    e0 = sim.observe()["E"]
    emax = 0.0
    for _ in range(10000):                      # 200 a.u.
        sim.step()
        emax = max(emax, abs(sim.observe()["E"] - e0))
    check("two particles: energy conservation", emax < 1e-3 * abs(e0),
          f"E0 = {e0:.5f} Ha, max drift = {emax:.2e} Ha over 200 a.u.")

    # 9. Two waves (mean field): norms and total energy conserved.
    sim.electrons[0].set_wave(pos1, mom1, 0.6)
    sim.electrons[1].set_wave(pos2, mom2, 0.6)
    e0 = sim.observe()["E"]
    emax = 0.0
    for _ in range(250):                        # 5 a.u.
        sim.step()
        emax = max(emax, abs(sim.observe()["E"] - e0))
    norms = [e.wave.expectations()["norm"] for e in sim.electrons]
    check("two waves: norm conservation", max(abs(n - 1) for n in norms) < 1e-10,
          f"norms = {norms[0]:.12f}, {norms[1]:.12f}")
    check("two waves: energy conservation", emax < 5e-3,
          f"E0 = {e0:.5f} Ha, max drift = {emax:.2e} Ha over 5 a.u.")

    # 10. Particle + wave: runs, stays finite.
    sim.electrons[1].set_particle(pos2, mom2)
    for _ in range(250):
        sim.step()
    c = sim.electrons[1].particle
    finite = all(map(math.isfinite, (c.x, c.y, c.px, c.py))) and np.isfinite(sim.electrons[0].wave.psi).all()
    check("particle + wave", finite, f"particle at ({c.x:.2f}, {c.y:.2f}), E = {sim.observe()['E']:.4f} Ha")

    # 11. Mean-field ground state: both in the same symmetric orbital, repulsion raises E.
    def mean_field_ground(sim, seeds):
        for e, seed in zip(sim.electrons, seeds):
            e.set_wave((0.0, 0.0), (0.0, 0.0), 1.0)
            e.wave.psi = seed
        e_old, e = np.inf, sim.observe()["E"]
        steps = 0
        while abs(e - e_old) > 1e-9 and steps < RELAX_MAX_STEPS:
            sim.relax([0, 1], 20)
            steps += 20
            e_old, e = e, sim.observe()["E"]
        return e, steps

    e2, steps = mean_field_ground(sim, [q.lcao(+1), q.lcao(+1)])
    r1, r2 = (e.density() for e in sim.electrons)
    same = np.max(np.abs(r1 - r2)) / np.max(r1)
    sym = np.max(np.abs(r1 - flip(r1))) / np.max(r1)
    vee = sim.observe()["Vee"]
    check("two-electron ground state (R = 2)", same < 1e-6 and sym < 1e-6 and e2 > 2 * eg,
          f"E = {e2:.5f} Ha (2 x E_sigma = {2 * eg:.5f}, V_ee = {vee:.4f}), "
          f"densities differ {same:.1e}, asymmetry {sym:.1e}, {steps} steps")

    # The faster shared-orbital relaxation (used by the app) must land on the same state.
    sim.electrons[0].wave.psi = q.gaussian((1.0, 0.5), (0, 0), 0.8)
    e_old, e_sh, steps_sh = np.inf, sim.observe()["E"], 0
    while abs(e_sh - e_old) > 1e-9 and steps_sh < RELAX_MAX_STEPS:
        sim.relax_shared(20)
        steps_sh += 20
        e_old, e_sh = e_sh, sim.observe()["E"]
    check("shared-orbital relaxation", abs(e_sh - e2) < 1e-6, f"E = {e_sh:.5f} Ha in {steps_sh} steps")

    # 12. Stretched bond: mean-field electrons lower their energy by moving to separate atoms.
    s3 = m.TwoCenterSystem(3.0, 1.0, 1.0, p["soft"])
    sim3 = m.Simulation(s3, absorb=False)
    sim3.count = 2
    q3 = sim3.electrons[0].wave
    e_sym, _ = mean_field_ground(sim3, [q3.lcao(+1), q3.lcao(+1)])
    e_brk, _ = mean_field_ground(sim3, [q3.gaussian((-1.5, 0), (0, 0), 0.7),
                                        q3.gaussian((1.5, 0), (0, 0), 0.7)])
    xs = [inf["x"] for inf in sim3.observe()["electrons"]]
    check("stretched bond (R = 3) breaks symmetry", e_brk < e_sym - 1e-3 and xs[0] < -0.5 < 0.5 < xs[1],
          f"shared sigma E = {e_sym:.4f} Ha, one electron per atom E = {e_brk:.4f} Ha, "
          f"<x> = {xs[0]:.2f}, {xs[1]:.2f}")

    print("-- moving nuclei (vibration) --")
    # 13. Force on a nucleus = minus the slope of the energy (Hellmann-Feynman).
    sh = m.TwoCenterSystem(2.0, 1.0, 1.0, p["soft"])
    qh = m.QuantumElectron(sh, absorb=False)
    h = 0.02
    e_side = []
    for x2 in (1.0 - h, 1.0 + h):
        sh.set_positions(-1.0, x2)
        e_side.append(qh.solve_state(qh.lcao(+1), tol=1e-11)[1] + sh.nuclear_repulsion())
    sh.set_positions(-1.0, 1.0)
    gs, _ = qh.solve_state(qh.lcao(+1), tol=1e-11)
    f_num = -(e_side[1] - e_side[0]) / (2 * h)
    f_hf = sh.nuclear_forces_density(np.abs(gs) ** 2)[1] + sh.nn_forces()[1]
    check("force on nucleus = -dE/dR", abs(f_hf - f_num) < 1e-3,
          f"from the charge cloud {f_hf:+.5f}, from the energy slope {f_num:+.5f} Ha/bohr")

    # 14. Two particles + moving nuclei (real H mass): total energy conserved.
    sv = m.TwoCenterSystem(1.6, 1.0, 1.0, p["soft"])
    simv = m.Simulation(sv, absorb=False)
    simv.count = 2
    simv.nuclei_move = True
    simv.electrons[0].set_particle(pos1, mom1)
    simv.electrons[1].set_particle(pos2, mom2)
    e0 = simv.observe()["E"]
    emax, rs = 0.0, []
    for _ in range(10000):                      # 200 a.u.
        simv.step()
        emax = max(emax, abs(simv.observe()["E"] - e0))
        rs.append(sv.R)
    check("vibrating nuclei + two particles: energy conservation", emax < 1e-3 * abs(e0),
          f"E0 = {e0:.5f} Ha, max drift = {emax:.2e} Ha, R between {min(rs):.3f} and {max(rs):.3f} bohr")

    # 15. One wave in sigma + moving nuclei (lighter mass): energy conserved, the
    #     bond starts to stretch toward its equilibrium length (> 2 bohr here).
    sv = m.TwoCenterSystem(2.0, 1.0, 1.0, p["soft"])
    simv = m.Simulation(sv, absorb=False)
    simv.nuclei_move = True
    simv.nuclear_mass = 200.0
    qv = simv.electrons[0].wave
    simv.electrons[0].mode = "wave"
    qv.psi = qv.solve_state(qv.lcao(+1))[0]
    e0 = simv.observe()["E"]
    emax = 0.0
    for _ in range(1000):                       # 20 a.u.
        simv.step()
        emax = max(emax, abs(simv.observe()["E"] - e0))
    check("vibrating nuclei + wave: energy conservation and stretch", emax < 2e-3 and sv.R > 2.02,
          f"E0 = {e0:.5f} Ha, max drift = {emax:.2e} Ha, R: 2.000 -> {sv.R:.3f} bohr")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_selftest() else 1)
