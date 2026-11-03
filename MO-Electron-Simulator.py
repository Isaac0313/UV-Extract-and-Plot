"""
MO Electron Simulator
---------------------
One or two electrons moving between two nuclei (H2+ / H2-like molecule).
Each electron can be simulated in two ways, and you can switch at any moment:

  PARTICLE  a point charge pushed around by the Coulomb forces of both
            nuclei (and of the other electron), Newton's laws F = -grad V.
  WAVE      a wavefunction psi(x, y, t), evolved with the time-dependent
            Schrodinger equation in the same potential.

Switching (acts on electron 1, electron 2 or both - "Act on"):
  Become wave      particle at (r, p)  ->  Gaussian wave packet centred on r,
                   moving with mean momentum p (width = "packet width" slider).
  Measure          wave -> particle. Position is drawn from |psi(r)|^2 and
                   momentum from |psi(k)|^2, so on average the particle gets
                   the same energy the wave had.
  Relax -> sigma   imaginary-time evolution: the wave "cools" into the lowest
                   orbital. With two electrons acting on both, they settle
                   together into the shared bonding orbital (mean field).
  Relax -> sigma*  same, but the lowest orbital is projected out, so the wave
                   settles into the next one (antibonding sigma*).
  Auto switch      random measure / become-wave events at a chosen rate.
  Ghost            (one electron) while the electron is a wave, a classical
                   "ghost" starts from the same point to compare the motions.
  Classical cloud  time-average of where a particle has been.

Does the electron stay in the orbital?
  The dashed white line encloses the region holding 90 % of the bonding
  (sigma) orbital of the bare molecule. The bottom-right plot shows, for each
  electron, whether it is inside that region (particle: 1 or 0) or how much
  of it is inside (wave: probability), and the status bar gives the average.

Two electrons:
  * particle + particle: two point charges repelling each other.
  * particle + wave: the particle feels the wave's charge cloud, the wave
    feels the particle's point charge.
  * wave + wave: each wave feels the other's charge cloud (time-dependent
    Hartree, i.e. mean-field). This keeps the cost at two 2D grids instead of
    one 4D grid, at the price of ignoring electron correlation. The
    electrons are taken to have opposite spins, so both may sit in sigma.

Mouse: click in the molecule plot to put an electron there; click-and-drag
to also give it a velocity. Waves stay waves, particles stay particles.

Setup (once):
    python -m pip install numpy matplotlib
    tkinter ships with the python.org installers (Linux: sudo apt install python3-tk)

Run:
    python MO-Electron-Simulator.py
    python MO-Electron-Simulator.py --selftest     (physics checks, no window)

Units: atomic units everywhere (hbar = m_e = e = 4*pi*eps0 = 1)
    length  1 bohr    = 0.529 Angstrom
    energy  1 hartree = 27.211 eV
    time    1 a.u.    = 0.02419 fs

Model assumptions (keep them in mind when interpreting results):
  * 2D: the electrons move in the plane that contains both nuclei.
  * Soft-core Coulomb interactions 1 / sqrt(d^2 + a^2), for electron-nucleus
    and electron-electron alike. The softening a removes the 1/r singularity
    so both the grid and Newton's equations stay well behaved; a -> 0 is the
    true Coulomb potential. Because of the 2D geometry and the softening,
    energies differ from real 3D H2+ / H2.
  * Nuclei are fixed (Born-Oppenheimer). Reported energies are electronic;
    the nucleus-nucleus repulsion Z1*Z2/R is shown separately.
  * Absorbing edges (optional) remove the part of a wave that leaves the
    box, so its norm is the probability that the electron is still bound.
"""

import math
import sys
import time
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Settings. Edit these to change the simulation; most can also be changed
# from inside the program with the sliders.
# ----------------------------------------------------------------------------
APP_TITLE = "MO Electron Simulator"
APP_SUBTITLE = "Electrons between two nuclei: particle ⇄ wavefunction"
HEADER_COLOR = "#1f3b57"
HEADER_TEXT_COLOR = "#ffffff"

GRID_N = 256              # grid points per side for each wavefunction
BOX_SIZE = 40.0           # box side length, bohr (grid spacing = BOX_SIZE / GRID_N)
VIEW_HALF_WIDTH = 10.0    # the molecule plot shows -10..10 bohr
DT = 0.02                 # time step, a.u. (particles and waves)
IMAG_DT = 0.05            # imaginary time step used by "Relax"
CLASSICAL_SUBSTEPS = 10   # Newton steps per DT (keeps energy conserved near the nuclei)
ABSORBER_WIDTH = 4.0      # width of the absorbing border, bohr
RELAX_STEPS_PER_FRAME = 20      # one electron
RELAX_STEPS_PER_FRAME_2E = 8    # two electrons (each step is more work)
RELAX_STEPS_PER_FRAME_SHARED = 15   # two electrons sharing one orbital
ESCAPE_DISTANCE = BOX_SIZE / 2 - ABSORBER_WIDTH - 2.0   # beyond this an electron counts as escaped
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
)
START = [((0.0, 1.2), (0.75, 0.0)),      # electron 1: position (bohr), momentum (a.u.)
         ((0.0, -1.2), (-0.75, 0.0))]    # electron 2
DRAG_TO_MOMENTUM = 0.4    # momentum per bohr of mouse drag

ELECTRON_COLORS = ("#4dabf7", "#ff922b")       # electrons on the dark molecule plot
ELECTRON_LINE_COLORS = ("#1971c2", "#e8590c")  # electrons on the white plots

FRAME_MS = 30             # target time between redraws, ms
TRAIL_LEN = 300
HISTORY_LEN = 1500
DISPLAY_GAMMA = 0.5       # brightness = (density / max)^gamma, < 1 shows faint tails
CLOUD_BLUR = 0.3          # smoothing of the classical cloud picture, bohr

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
        self.version = 0
        self.set_params(R, Z1, Z2, soft)

    def set_params(self, R, Z1, Z2, soft):
        self.R, self.Z1, self.Z2, self.soft = float(R), float(Z1), float(Z2), float(soft)
        self.nuclei = [(-self.R / 2, self.Z1), (self.R / 2, self.Z2)]
        self.V = self.potential(self.X, self.Y)
        # Electron-electron kernel 1/sqrt(d^2 + a^2) in k space, for the
        # Hartree potential of a charge cloud (exact for |r - r'| < box / 2).
        kernel = 1.0 / np.sqrt(self._DX ** 2 + self._DY ** 2 + self.soft ** 2)
        self.ee_kernel_k = np.fft.rfft2(kernel) * self.dA
        self.version += 1

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
        return self.Z1 * self.Z2 / self.R

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
        """Call after the potential changes."""
        s = self.sys
        self.half_v = np.exp(-0.5j * DT * s.V)
        self.kin = np.exp(-1j * DT * s.T)
        self.half_v_im = np.exp(-0.5 * IMAG_DT * s.V)
        self.kin_im = np.exp(-IMAG_DT * s.T)

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

    def __init__(self, system, absorb=True):
        self.sys = system
        self.mode = "particle"
        self.particle = None
        self.wave = QuantumElectron(system, absorb)

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

    def __init__(self, system, absorb=True):
        self.sys = system
        self.electrons = [Electron(system, absorb), Electron(system, absorb)]
        self.count = 1
        self.t = 0.0

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

    def step(self, clouds=(None, None)):
        """Advance all active electrons by DT."""
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
        self.t += DT

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
        return dict(electrons=infos, T=t_sum, V=vn_sum + vee, Vee=vee, E=t_sum + vn_sum + vee)


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
def run_selftest():
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    p = DEFAULTS
    s = TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
    (pos1, mom1), (pos2, mom2) = START

    print("-- one electron --")
    # 1. Particle: total energy conserved by velocity Verlet.
    c = ClassicalElectron(s, pos1, mom1)
    e0 = c.energies()[2]
    emax = 0.0
    for _ in range(20000):                      # 400 a.u. of time
        c.step(DT)
        emax = max(emax, abs(c.energies()[2] - e0))
    check("classical energy conservation", emax < 1e-3 * abs(e0),
          f"E0 = {e0:.6f} Ha, max drift = {emax:.2e} Ha over 400 a.u.")

    # 2. Wave: norm conserved without the absorber.
    q = QuantumElectron(s, absorb=False)
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
    mask, _ = orbital_region(g, s.dA)
    share = float(np.sum(np.abs(g) ** 2 * mask) * s.dA)
    check("orbital region", abs(share - ORBITAL_REGION_FRACTION) < 0.01, f"sigma inside = {share:.3f}")

    print("-- two electrons --")
    # 7. Hartree potential of a small charge cloud looks like a point charge from afar.
    q.from_particle((0.0, 0.0), (0.0, 0.0), 0.3)
    vh = s.hartree(np.abs(q.psi) ** 2)
    got, want = s.interp(vh, 5.0, 0.0), 1.0 / math.sqrt(25 + s.soft ** 2)
    check("Hartree potential", abs(got / want - 1) < 0.02, f"V_H(5, 0) = {got:.4f}, point charge {want:.4f}")

    # 8. Two particles: total energy (including their repulsion) conserved.
    sim = Simulation(s, absorb=False)
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
    s3 = TwoCenterSystem(3.0, 1.0, 1.0, p["soft"])
    sim3 = Simulation(s3, absorb=False)
    sim3.count = 2
    q3 = sim3.electrons[0].wave
    e_sym, _ = mean_field_ground(sim3, [q3.lcao(+1), q3.lcao(+1)])
    e_brk, _ = mean_field_ground(sim3, [q3.gaussian((-1.5, 0), (0, 0), 0.7),
                                        q3.gaussian((1.5, 0), (0, 0), 0.7)])
    xs = [inf["x"] for inf in sim3.observe()["electrons"]]
    check("stretched bond (R = 3) breaks symmetry", e_brk < e_sym - 1e-3 and xs[0] < -0.5 < 0.5 < xs[1],
          f"shared sigma E = {e_sym:.4f} Ha, one electron per atom E = {e_brk:.4f} Ha, "
          f"<x> = {xs[0]:.2f}, {xs[1]:.2f}")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


# ============================================================================
# User interface
# ============================================================================
def run_app(root=None, mainloop=True):
    """Open the simulator window. Pass mainloop=False to get the app object back
    without blocking (used for scripted tests / screenshots)."""
    import tkinter as tk
    from tkinter import ttk
    from matplotlib import colormaps
    from matplotlib.colors import hsv_to_rgb, to_rgb
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:                                    # sharper text on Windows high-DPI screens
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    electron_rgb = [np.array(to_rgb(c)) for c in ELECTRON_COLORS]

    class SimulatorApp:
        def __init__(self, root):
            self.root = root
            root.title(APP_TITLE)
            self.rng = np.random.default_rng()
            p = DEFAULTS
            self.sys = TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
            self.sim = Simulation(self.sys)
            self.cmap = colormaps["inferno"]
            self.blur_k = np.exp(-0.5 * CLOUD_BLUR ** 2 * (self.sys.KX ** 2 + self.sys.KY ** 2))
            self.orbitals = {}              # {0: (psi, E), 1: (psi, E)} one-electron orbitals
            self.orbital_version = -1
            self.region = None              # (mask, threshold) of the sigma orbital
            self.region_art = None
            self.region_dirty_since = None
            self.running = True
            self.drag_start = None
            self.last_dt = 0.0
            self.inside_acc = [[0.0, 0.0], [0.0, 0.0]]   # [time inside, total time] per electron
            self._build_ui()
            self.reset()
            self.root.after(FRAME_MS, self.tick)

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        def _build_ui(self):
            root = self.root
            header = tk.Frame(root, bg=HEADER_COLOR)
            header.pack(fill="x")
            tk.Label(header, text=APP_TITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR,
                     font=("TkDefaultFont", 14, "bold")).pack(side="left", padx=12, pady=6)
            tk.Label(header, text=APP_SUBTITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR
                     ).pack(side="left", padx=6, pady=6)

            # Status bar and control panel are packed before the plot so they keep
            # their space when the window is small; the plot takes what is left.
            self.status = tk.StringVar()
            tk.Label(root, textvariable=self.status, anchor="w", justify="left",
                     font=("TkFixedFont", 9), relief="sunken", padx=6
                     ).pack(fill="x", side="bottom")

            body = tk.Frame(root)
            body.pack(fill="both", expand=True)
            panel = tk.Frame(body, padx=6, pady=2)
            panel.pack(side="right", fill="y")

            # --- figure -------------------------------------------------------
            self.fig = Figure(figsize=(9.6, 6.4), dpi=100)
            gs = self.fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.62, wspace=0.25,
                                       left=0.075, right=0.98, top=0.94, bottom=0.08)
            self.ax = self.fig.add_subplot(gs[:, 0])
            self.ax_cut = self.fig.add_subplot(gs[0, 1])
            self.ax_e = self.fig.add_subplot(gs[1, 1])
            self.ax_in = self.fig.add_subplot(gs[2, 1], sharex=self.ax_e)
            self.canvas = FigureCanvasTkAgg(self.fig, master=body)
            self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
            self._build_axes()
            self.canvas.mpl_connect("button_press_event", self.on_press)
            self.canvas.mpl_connect("motion_notify_event", self.on_motion)
            self.canvas.mpl_connect("button_release_event", self.on_release)

            # --- controls -----------------------------------------------------
            box = ttk.LabelFrame(panel, text="Electrons")
            box.pack(fill="x", pady=3)
            self.mode_var = tk.StringVar()
            tk.Label(box, textvariable=self.mode_var, font=("TkDefaultFont", 10, "bold"),
                     justify="left").grid(row=0, column=0, columnspan=2, sticky="w", padx=4)
            row = tk.Frame(box)
            row.grid(row=1, column=0, columnspan=2, sticky="w", padx=2)
            tk.Label(row, text="Number:").pack(side="left")
            self.count_var = tk.IntVar(value=1)
            for n in (1, 2):
                ttk.Radiobutton(row, text=str(n), value=n, variable=self.count_var,
                                command=self.on_count).pack(side="left")
            row = tk.Frame(box)
            row.grid(row=2, column=0, columnspan=2, sticky="w", padx=2)
            tk.Label(row, text="Act on:").pack(side="left")
            self.target_var = tk.StringVar(value="1")
            self.target_buttons = []
            for val, text in (("1", "e₁"), ("2", "e₂"), ("both", "both")):
                b = ttk.Radiobutton(row, text=text, value=val, variable=self.target_var)
                b.pack(side="left")
                b.state(["disabled"])                 # enabled when there are two electrons
                self.target_buttons.append(b)
            buttons = [
                ("Become wave ψ", self.become_wave),
                ("Measure (collapse)", self.measure),
                ("Relax → σ bonding", lambda: self.start_relax(0)),
                ("Relax → σ* anti", lambda: self.start_relax(1)),
            ]
            for k, (text, cmd) in enumerate(buttons):
                ttk.Button(box, text=text, command=cmd).grid(
                    row=3 + k // 2, column=k % 2, sticky="ew", padx=2, pady=1)
            self.play_btn = ttk.Button(box, text="Pause", command=self.toggle_run)
            self.play_btn.grid(row=5, column=0, sticky="ew", padx=2, pady=1)
            ttk.Button(box, text="Reset", command=self.reset).grid(
                row=5, column=1, sticky="ew", padx=2, pady=1)

            self.vars = {}
            box = ttk.LabelFrame(panel, text="Molecule")
            box.pack(fill="x", pady=3)
            self._slider(box, "R", "Bond length R (bohr)", 0.4, 8.0, 0.05, self.on_molecule)
            self._slider(box, "Z1", "Left nucleus charge Z₁", 0.5, 3.0, 0.05, self.on_molecule)
            self._slider(box, "Z2", "Right nucleus charge Z₂", 0.5, 3.0, 0.05, self.on_molecule)
            self._slider(box, "soft", "Soft-core a (bohr)", 0.2, 1.5, 0.05, self.on_molecule)

            box = ttk.LabelFrame(panel, text="Simulation")
            box.pack(fill="x", pady=3)
            self._slider(box, "sigma", "Wave packet width (bohr)", 0.2, 2.0, 0.05, None)
            self._slider(box, "steps", "Speed (steps / frame)", 1, 25, 1, None)
            self._slider(box, "rate", "Auto switch rate (1/a.u.)", 0.005, 0.5, 0.005, None)

            box = ttk.LabelFrame(panel, text="Options")
            box.pack(fill="x", pady=3)
            self.flags = {}
            for key, text, val in [("auto", "Auto switch (random)", False),
                                   ("ghost", "Classical ghost (1 electron)", True),
                                   ("cloud", "Classical cloud (time average)", False),
                                   ("absorb", "Absorbing edges (ionization)", True),
                                   ("phase", "Show phase (1 electron)", False)]:
                v = tk.BooleanVar(value=val)
                ttk.Checkbutton(box, text=text, variable=v, command=self.on_flag).pack(anchor="w")
                self.flags[key] = v

        def _slider(self, parent, key, label, lo, hi, res, command):
            var = tk.DoubleVar(value=DEFAULTS[key])
            self.vars[key] = var
            row = tk.Frame(parent)
            row.pack(fill="x", padx=4)
            tk.Label(row, text=label).pack(side="left")
            tk.Label(row, textvariable=var, width=6, anchor="e").pack(side="right")
            tk.Scale(parent, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var,
                     showvalue=0, width=10, sliderlength=18, length=220,
                     command=(lambda _v: command()) if command else None
                     ).pack(fill="x", padx=4, pady=(0, 2))

        def _build_axes(self):
            s = self.sys
            ax = self.ax
            # Only the visible part of the grid is turned into an image (faster).
            i0 = int(np.searchsorted(s.x, -VIEW_HALF_WIDTH - 2 * s.dx))
            self.view = slice(i0, s.n - i0)
            h = s.dx / 2
            xv = s.x[self.view]
            ext = (xv[0] - h, xv[-1] + h, xv[0] - h, xv[-1] + h)
            self.img = ax.imshow(np.zeros((xv.size, xv.size, 3), np.uint8), origin="lower",
                                 extent=ext, interpolation="bilinear")
            ax.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            ax.set_ylim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            ax.set_xlabel("x (bohr)")
            ax.set_ylabel("y (bohr)")
            self.title = ax.set_title("")
            self.contours = None
            self.nuclei_art = ax.scatter([], [], s=[], c="#e8483b", edgecolors="white",
                                         linewidths=1.2, zorder=5)
            self.trails, self.dots, self.trail_pts = [], [], []
            for col in ELECTRON_COLORS:
                (tr,) = ax.plot([], [], "-", color=col, lw=1.2, alpha=0.85, zorder=6)
                (dot,) = ax.plot([], [], "o", color=col, ms=9, mec="white", zorder=7)
                self.trails.append(tr)
                self.dots.append(dot)
                self.trail_pts.append(deque(maxlen=TRAIL_LEN))
            (self.ghost_trail,) = ax.plot([], [], "--", color="white", lw=0.9, alpha=0.6, zorder=6)
            (self.ghost_dot,) = ax.plot([], [], "o", mfc="none", mec="white", ms=9, zorder=7)
            self.ghost_pts = deque(maxlen=TRAIL_LEN)
            (self.drag_line,) = ax.plot([], [], "-", color="#ffe066", lw=2, zorder=8)

            a = self.ax_cut
            a.set_xlabel("x along the bond, cut at y = 0 (bohr)", fontsize=9)
            a.set_ylabel("energy (Ha)")
            (self.cut_v,) = a.plot(s.x, s.V[s.n // 2], color="#555", lw=1.5, label="V(x, 0)")
            self.cut_rhos, self.cut_dots = [], []
            for k, col in enumerate(ELECTRON_LINE_COLORS):
                (line,) = a.plot([], [], color=col, lw=1.5,
                                 label="|ψ(x,0)|² (scaled)" if k == 0 else None)
                (dot,) = a.plot([], [], "o", color=col, ms=7)
                self.cut_rhos.append(line)
                self.cut_dots.append(dot)
            (self.cut_e,) = a.plot([], [], ":", color="#333", lw=1.2, label="electron energy")
            a.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_e
            a.set_ylabel("energy (Ha)")
            a.tick_params(labelbottom=False)
            (self.l_t,) = a.plot([], [], color="#2f9e44", lw=1.2, label="kinetic T")
            (self.l_v,) = a.plot([], [], color="#7048e8", lw=1.2, label="potential V")
            (self.l_e,) = a.plot([], [], color="black", lw=1.6, label="total E")
            self.level_lines = [a.axhline(np.nan, color=c, ls="--", lw=0.9)
                                for c in ("#1c7ed6", "#e8590c")]
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_in
            a.set_xlabel("time (fs)")
            a.set_ylabel("inside σ")
            a.set_ylim(-0.05, 1.08)
            self.l_in = [a.plot([], [], color=col, lw=1.3, label=f"electron {k + 1}")[0]
                         for k, col in enumerate(ELECTRON_LINE_COLORS)]
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            # Blitting: everything that changes every frame is "animated" and redrawn
            # on top of a cached picture of the static parts (axes, ticks, labels).
            self.anim = [self.nuclei_art, *self.trails, self.ghost_trail, *self.dots,
                         self.ghost_dot, self.drag_line, self.title,
                         *self.cut_rhos, *self.cut_dots, self.cut_e,
                         self.l_t, self.l_v, self.l_e, *self.level_lines, *self.l_in]
            for art in [self.img, *self.anim]:
                art.set_animated(True)
            self.background = None
            self.need_full_draw = True
            self.canvas.mpl_connect("draw_event", self._on_draw)

        def _on_draw(self, _event):
            self.background = self.canvas.copy_from_bbox(self.fig.bbox)
            self._draw_animated()

        def _draw_animated(self):
            # image first, then potential contours and the orbital boundary on top of it
            for art in [self.img, self.contours, self.region_art, *self.anim]:
                if art is not None:
                    self.fig.draw_artist(art)

        def _refresh_canvas(self):
            if self.background is None or self.need_full_draw:
                self.need_full_draw = False
                self.canvas.draw()                  # also re-caches the background
            else:
                self.canvas.restore_region(self.background)
                self._draw_animated()
                self.canvas.blit(self.fig.bbox)

        def _update_energy_limits(self, h):
            """Rescale the energy plot only when the data leaves it (rescaling = full redraw)."""
            ax = self.ax_e
            t0, t1 = h[0, 0], h[-1, 0]
            vals = h[:, 1:4]
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            if self.sim.count == 1:
                for _psi, e in self._orbitals().values():
                    lo, hi = min(lo, e), max(hi, e)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            span = max(hi - lo, 0.2)
            if t1 > x1 or t0 < x0:
                ax.set_xlim(t0, t0 + max(1.5 * (t1 - t0), 0.01))
                self.need_full_draw = True
            if lo < y0 or hi > y1 or span < 0.3 * (y1 - y0):
                ax.set_ylim(lo - 0.15 * span, hi + 0.35 * span)
                self.need_full_draw = True

        def _draw_molecule(self):
            s = self.sys
            if self.contours is not None:
                self.contours.remove()
            levels = np.linspace(s.V.min() * 0.9, -0.15, 9)
            self.contours = self.ax.contour(s.X, s.Y, s.V, levels=levels, colors="white",
                                            linewidths=0.5, alpha=0.3)
            self.contours.set_animated(True)
            self.nuclei_art.set_offsets([(xn, 0) for xn, _ in s.nuclei])
            self.nuclei_art.set_sizes([60 + 60 * z for _, z in s.nuclei])
            self.cut_v.set_ydata(s.V[s.n // 2])
            self.ax_cut.set_ylim(s.V.min() - 0.3, 1.0)
            self.need_full_draw = True

        def _orbitals(self):
            if self.orbital_version != self.sys.version:
                self.orbitals = {}
                self.region = None
                self.orbital_version = self.sys.version
                if self.region_art is not None:
                    self.region_art.remove()
                    self.region_art = None
            return self.orbitals

        def _ensure_sigma(self):
            """One-electron sigma orbital of the current molecule and its 90 % region."""
            orb = self._orbitals()
            if 0 not in orb:
                self.status.set("Computing the σ orbital of this molecule ...")
                self.root.update_idletasks()
                q = self.sim.electrons[0].wave
                orb[0] = q.solve_state(q.lcao(+1))
            if self.region is None:
                s = self.sys
                psi = orb[0][0]
                self.region = orbital_region(psi, s.dA)
                self.region_art = self.ax.contour(s.X, s.Y, np.abs(psi) ** 2, levels=[self.region[1]],
                                                  colors="white", linewidths=1.0,
                                                  linestyles="--", alpha=0.8)
                self.region_art.set_animated(True)
            return orb[0]

        # ------------------------------------------------------------------
        # State changes
        # ------------------------------------------------------------------
        def reset(self):
            p = DEFAULTS
            for k, v in self.vars.items():
                v.set(p[k])
            self.sys.set_params(p["R"], p["Z1"], p["Z2"], p["soft"])
            self.sim.rebuild()
            self.sim.set_absorb(self.flags["absorb"].get())
            self.sim.t = 0.0
            self.history = deque(maxlen=HISTORY_LEN)
            self.clouds = [np.zeros((self.sys.n, self.sys.n)) for _ in range(2)]
            self.relaxing = None
            self.ghost = None
            self.ghost_pts.clear()
            for i in range(2):
                self.place(i, *START[i], as_wave=False)
            self._draw_molecule()
            self._ensure_sigma()
            self.message = ("Particle mode: point charges pulled by the nuclei. Click the plot to place "
                            "an electron, drag to give it a velocity.")

        def targets(self):
            if self.sim.count == 1:
                return [0]
            return {"1": [0], "2": [1], "both": [0, 1]}[self.target_var.get()]

        def place(self, i, pos, mom, as_wave):
            e = self.sim.electrons[i]
            if as_wave:
                e.set_wave(pos, mom, self.vars["sigma"].get())
            else:
                e.set_particle(pos, mom)
            self.trail_pts[i].clear()
            self.inside_acc[i] = [0.0, 0.0]
            self.relaxing = None
            if i == 0:
                self.ghost = None
                self.ghost_pts.clear()
                if as_wave and self.sim.count == 1 and self.flags["ghost"].get():
                    self.ghost = ClassicalElectron(self.sys, pos, mom)

        def _to_wave(self, i, for_relax=False):
            """Particle -> wave packet. An electron that has escaped the molecule is
            refused (a packet at the box edge would wrap around), unless we are
            about to relax it anyway, in which case it restarts at the centre."""
            c = self.sim.electrons[i].particle
            if max(abs(c.x), abs(c.y)) > ESCAPE_DISTANCE:
                if not for_relax:
                    return False
                self.place(i, (0.0, 0.0), (0.0, 0.0), as_wave=True)
                return True
            self.place(i, (c.x, c.y), (c.px, c.py), as_wave=True)
            return True

        def become_wave(self):
            idxs = [i for i in self.targets() if self.sim.electrons[i].mode == "particle"]
            if not idxs:
                self.message = "Already a wave. Use Measure to turn it back into a particle."
                return
            e_before = self.sim.observe()["E"]
            done = [i for i in idxs if self._to_wave(i)]
            escaped = [i for i in idxs if i not in done]
            msg = []
            if done:
                e_after = self.sim.observe()["E"]
                if self.sim.count == 1:
                    who = "The electron became a wave packet"
                elif len(done) == 2:
                    who = "Both electrons became wave packets"
                else:
                    who = f"Electron {done[0] + 1} became a wave packet"
                msg.append(f"{who} (same <r>, <p>). Energy {e_before:.3f} → {e_after:.3f} Ha: "
                           f"squeezing an electron into width σ costs kinetic energy (uncertainty principle).")
            for i in escaped:
                label = f"Electron {i + 1}" if self.sim.count == 2 else "The electron"
                msg.append(f"{label} has escaped the molecule (ionized), so it stays a particle. "
                           f"Click to place it again or press Reset.")
            self.message = " ".join(msg)

        def measure(self):
            idxs = [i for i in self.targets() if self.sim.electrons[i].mode == "wave"]
            if not idxs:
                self.message = "Already a particle. Use Become wave first."
                return
            parts = []
            for i in idxs:
                pos, mom = self.sim.electrons[i].wave.measure(self.rng)
                self.place(i, pos, mom, as_wave=False)
                e = self.sim.electrons[i].particle.energies()[2]
                label = f"electron {i + 1}" if self.sim.count == 2 else "electron"
                parts.append(f"{label} found at ({pos[0]:.2f}, {pos[1]:.2f}), one-body energy {e:.3f} Ha"
                             + (" (unbound, it will escape)" if e > 0 and self.sim.count == 1 else ""))
            self.message = "Measured! " + "; ".join(parts)

        def start_relax(self, target):
            idxs = self.targets()
            if target == 1 and len(idxs) == 2:
                self.message = "Choose one electron (Act on: e₁ or e₂) to relax into σ*."
                return
            for i in idxs:
                if self.sim.electrons[i].mode == "particle":
                    self._to_wave(i, for_relax=True)
            if target == 0 and len(idxs) == 2:
                # Both into sigma: relax one orbital that the two electrons share.
                qa, qb = (e.wave for e in self.sim.electrons)
                phi = qa.normalized(qa.normalized(qa.psi) + qa.normalized(qb.psi) + 0.1 * qa.lcao(+1))
                qa.psi, qb.psi = phi, phi.copy()
                self.relaxing = dict(idxs=idxs, project={}, target=0, e_old=np.inf, steps=0, shared=True)
                self.ghost = None
                self.ghost_pts.clear()
                self.message = "Relaxing (imaginary time): both electrons settle into one shared σ orbital ..."
                return
            project = {}
            for i in idxs:
                e = self.sim.electrons[i]
                q = e.wave
                if target == 1:
                    other = self.sim.other(e)
                    if other is None:
                        g = self._ensure_sigma()[0]
                    else:
                        # lowest orbital in the field of the (frozen) other electron
                        self.status.set("Finding the lowest orbital first ...")
                        self.root.update_idletasks()
                        g = q.solve_state(q.lcao(+1), v_extra=self.sim.field_of(other))[0]
                    project[i] = [g]
                # A little LCAO guess makes sure the target orbital is present in the start state.
                psi = q.normalized(q.psi) + 0.05 * q.lcao(+1 if target == 0 else -1)
                for phi in project.get(i, ()):
                    psi = psi - phi * (np.vdot(phi, psi) * self.sys.dA)
                q.psi = q.normalized(psi)
            self.relaxing = dict(idxs=idxs, project=project, target=target, e_old=np.inf, steps=0,
                                 shared=False)
            self.ghost = None
            self.ghost_pts.clear()
            name = "σ bonding" if target == 0 else "σ* antibonding"
            self.message = f"Relaxing (imaginary time) into the {name} orbital ..."

        def _finish_relax(self, e_total, converged):
            r = self.relaxing
            self.relaxing = None
            name = "σ bonding" if r["target"] == 0 else "σ* antibonding"
            note = "" if converged else " (stopped before full convergence)"
            rep = self.sys.nuclear_repulsion()
            if self.sim.count == 1:
                q = self.sim.electrons[0].wave
                self._orbitals()[r["target"]] = (q.psi.copy(), e_total)
                extra = (f"; with nuclear repulsion {e_total + rep:.4f} Ha "
                         f"({'bound' if e_total + rep < 0 else 'not bound'})" if r["target"] == 0 else "")
                self.message = (f"Settled into the {name} orbital{note}: E = {e_total:.4f} Ha = "
                                f"{e_total * HARTREE_EV:.2f} eV{extra}. It is now a stationary state.")
            else:
                who = "Both electrons" if len(r["idxs"]) == 2 else f"Electron {r['idxs'][0] + 1}"
                where = "one shared " if r["shared"] else "the "
                self.message = (f"{who} settled into {where}{name} orbital (mean field){note}. "
                                f"Total electronic E = {e_total:.4f} Ha, + nuclear repulsion = "
                                f"{e_total + rep:.4f} Ha.")

        def on_count(self):
            n = self.count_var.get()
            if n == self.sim.count:
                return
            self.sim.count = n
            self.relaxing = None
            for b in self.target_buttons:
                b.state(["!disabled"] if n == 2 else ["disabled"])
            if n == 2:
                # Start the new electron at the mirror image of electron 1 (r -> -r, p -> -p).
                e1, e2 = self.sim.electrons
                if e1.mode == "particle":
                    c = e1.particle
                    self.place(1, (-c.x, -c.y), (-c.px, -c.py), as_wave=False)
                else:
                    self.place(1, (0.0, 0.0), (0.0, 0.0), as_wave=True)
                    e2.wave.psi = e1.wave.psi[::-1, ::-1].copy()
                self.ghost = None
                self.ghost_pts.clear()
                self.message = ("Second electron added at the mirror image of electron 1. Both feel the "
                                "nuclei and repel each other. 'Act on' picks which electron the buttons change.")
            else:
                self.message = "Back to one electron."
            self.history.clear()

        def on_molecule(self):
            v = self.vars
            self.sys.set_params(v["R"].get(), v["Z1"].get(), v["Z2"].get(), v["soft"].get())
            self.sim.rebuild()
            if self.ghost is not None:
                self.ghost.refresh_force()
            for c in self.clouds:
                c[:] = 0
            self._orbitals()                    # drops the old sigma and its boundary
            self.region_dirty_since = time.time()
            self._draw_molecule()

        def on_flag(self):
            self.sim.set_absorb(self.flags["absorb"].get())
            if not self.flags["cloud"].get():
                for c in self.clouds:
                    c[:] = 0
            if not self.flags["ghost"].get():
                self.ghost = None
                self.ghost_pts.clear()

        def toggle_run(self):
            self.running = not self.running
            self.play_btn.config(text="Pause" if self.running else "Play")

        # ------------------------------------------------------------------
        # Mouse: click = place, drag = place with velocity
        # ------------------------------------------------------------------
        def on_press(self, ev):
            if ev.inaxes is self.ax and ev.xdata is not None:
                self.drag_start = (ev.xdata, ev.ydata)

        def on_motion(self, ev):
            if self.drag_start and ev.inaxes is self.ax and ev.xdata is not None:
                x0, y0 = self.drag_start
                self.drag_line.set_data([x0, ev.xdata], [y0, ev.ydata])
                if not self.running:
                    self._refresh_canvas()

        def on_release(self, ev):
            if not self.drag_start:
                return
            x0, y0 = self.drag_start
            self.drag_start = None
            self.drag_line.set_data([], [])
            x1, y1 = (ev.xdata, ev.ydata) if ev.xdata is not None else (x0, y0)
            mom = (DRAG_TO_MOMENTUM * (x1 - x0), DRAG_TO_MOMENTUM * (y1 - y0))
            i = self.targets()[0]
            as_wave = self.sim.electrons[i].mode == "wave"
            self.place(i, (x0, y0), mom, as_wave=as_wave)
            label = f"electron {i + 1}" if self.sim.count == 2 else "electron"
            self.message = (f"Placed {label} ({'wave packet' if as_wave else 'particle'}) at "
                            f"({x0:.2f}, {y0:.2f}) with momentum ({mom[0]:.2f}, {mom[1]:.2f}).")
            self.render()

        # ------------------------------------------------------------------
        # Animation loop
        # ------------------------------------------------------------------
        def tick(self):
            if self.region_dirty_since and time.time() - self.region_dirty_since > 0.4 \
                    and self.drag_start is None:
                self.region_dirty_since = None
                self._ensure_sigma()
                self.need_full_draw = True
            if self.running:
                try:
                    self.advance()
                    self.render()
                except Exception as exc:           # keep the window alive
                    self.running = False
                    self.message = f"Error: {exc}"
                    self.play_btn.config(text="Play")
            self.root.after(FRAME_MS, self.tick)

        def advance(self):
            self.last_dt = 0.0
            if self.relaxing is not None:
                r = self.relaxing
                if r["shared"]:
                    n = RELAX_STEPS_PER_FRAME_SHARED
                    self.sim.relax_shared(n)
                else:
                    n = RELAX_STEPS_PER_FRAME if self.sim.count == 1 else RELAX_STEPS_PER_FRAME_2E
                    self.sim.relax(r["idxs"], n, r["project"])
                r["steps"] += n
                e = self.sim.observe()["E"]
                if abs(e - r["e_old"]) < RELAX_TOL or r["steps"] >= RELAX_MAX_STEPS:
                    self._finish_relax(e, abs(e - r["e_old"]) < RELAX_TOL)
                else:
                    r["e_old"] = e
                return

            steps = int(self.vars["steps"].get())
            clouds = self.clouds if self.flags["cloud"].get() else (None, None)
            for _ in range(steps):
                self.sim.step(clouds)
                if self.ghost is not None:
                    self.ghost.step(DT)
            self.last_dt = steps * DT

            if self.flags["auto"].get():
                if self.rng.random() < 1 - math.exp(-self.vars["rate"].get() * steps * DT):
                    i = int(self.rng.integers(self.sim.count))
                    if self.sim.electrons[i].mode == "particle":
                        if self._to_wave(i):
                            self.message = f"Auto: electron {i + 1} became a wave."
                    else:
                        pos, mom = self.sim.electrons[i].wave.measure(self.rng)
                        self.place(i, pos, mom, as_wave=False)
                        self.message = f"Auto: electron {i + 1} measured at ({pos[0]:.2f}, {pos[1]:.2f})."

        # ------------------------------------------------------------------
        # Drawing
        # ------------------------------------------------------------------
        def _brightness(self, rho):
            rho = rho[self.view, self.view]
            m = rho.max()
            return (rho / m) ** DISPLAY_GAMMA if m > 0 else rho

        def _smoothed_cloud(self, i):
            c = self.clouds[i]
            if not self.flags["cloud"].get() or c.max() == 0:
                return None
            return np.fft.ifft2(np.fft.fft2(c) * self.blur_k).real.clip(0)

        def _make_image(self, infos):
            if len(infos) == 1:
                info = infos[0]
                layer = info["rho"] if info["mode"] == "wave" else self._smoothed_cloud(0)
                if layer is None:
                    return np.zeros(self.img.get_array().shape, np.uint8)
                b = self._brightness(layer)
                if info["mode"] == "wave" and self.flags["phase"].get():
                    psi = self.sim.electrons[0].wave.psi[self.view, self.view]
                    hsv = np.stack([(np.angle(psi) + np.pi) / (2 * np.pi), np.ones_like(b), b], axis=-1)
                    return (hsv_to_rgb(hsv) * 255).astype(np.uint8)
                return self.cmap(b, bytes=True)[..., :3]
            rgb = np.zeros(self.img.get_array().shape)
            for i, info in enumerate(infos):     # one colour per electron, added together
                layer = info["rho"] if info["mode"] == "wave" else self._smoothed_cloud(i)
                if layer is not None:
                    rgb += self._brightness(layer)[..., None] * electron_rgb[i]
            return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        def _inside(self, info):
            if self.region is None:
                return np.nan
            mask = self.region[0]
            if info["mode"] == "particle":
                idx = self.sys.grid_index(info["x"], info["y"])
                return float(mask[idx]) if idx is not None else 0.0
            return float(np.sum(info["rho"] * mask) * self.sys.dA)

        def render(self):
            s = self.sys
            obs = self.sim.observe()
            infos = obs["electrons"]
            two = len(infos) == 2
            sigma = self._orbitals().get(0)

            self.img.set_data(self._make_image(infos))
            inside, lines = [np.nan, np.nan], []
            for i in range(2):
                if i >= len(infos):
                    self.dots[i].set_data([], [])
                    self.trails[i].set_data([], [])
                    self.cut_rhos[i].set_data([], [])
                    self.cut_dots[i].set_data([], [])
                    continue
                info = infos[i]
                x, y = info["x"], info["y"]
                is_wave = info["mode"] == "wave"
                self.dots[i].set_data([x], [y])
                self.dots[i].set_marker("P" if is_wave else "o")
                self.trail_pts[i].append((x, y))
                tx, ty = zip(*self.trail_pts[i])
                self.trails[i].set_data(tx, ty)

                ins = self._inside(info)
                inside[i] = ins
                acc = self.inside_acc[i]
                if self.last_dt > 0 and not np.isnan(ins):
                    acc[0] += ins * self.last_dt
                    acc[1] += self.last_dt
                avg = f" (avg {acc[0] / acc[1]:.0%})" if acc[1] > 0 else ""
                label = f"e{i + 1} " if two else ""
                if is_wave:
                    pop = ""
                    if sigma is not None:
                        p_sigma = abs(np.vdot(sigma[0], self.sim.electrons[i].wave.psi)) ** 2 * s.dA ** 2
                        pop = f"  P(σ) = {p_sigma:.2f}"
                    lines.append(f"{label}WAVE <r> = ({x:5.2f}, {y:5.2f})  in σ region {ins:4.0%}{avg}"
                                 f"{pop}  bound {info['norm']:.3f}")
                else:
                    lines.append(f"{label}PARTICLE r = ({x:5.2f}, {y:5.2f})  p = ({info['px']:5.2f}, "
                                 f"{info['py']:5.2f})  in σ region: {'yes' if ins >= 0.5 else 'no '}{avg}")

                # cut along the bond
                if two:
                    base = self.ax_cut.get_ylim()[0] + 0.1
                    row = info["rho"][s.n // 2] if is_wave else None
                    if row is not None and row.max() > 0:
                        self.cut_rhos[i].set_data(s.x, base + 1.2 * row / row.max())
                    else:
                        self.cut_rhos[i].set_data([], [])
                    self.cut_dots[i].set_data(([], []) if is_wave else ([x], [base]))
                else:
                    e1 = info["T"] + info["V"]
                    row = info["rho"][s.n // 2] if is_wave else None
                    if row is None:
                        cl = self._smoothed_cloud(0)
                        row = None if cl is None else cl[s.n // 2]
                    if row is not None and row.max() > 0:
                        self.cut_rhos[i].set_data(s.x, e1 + 0.8 * row / row.max())
                    else:
                        self.cut_rhos[i].set_data([], [])
                    self.cut_dots[i].set_data(([], []) if is_wave else ([x], [e1]))
            if two:
                self.cut_e.set_data([], [])
            else:
                e1 = infos[0]["T"] + infos[0]["V"]
                self.cut_e.set_data([s.x[0], s.x[-1]], [e1, e1])

            if self.ghost is not None and not two:
                g = self.ghost
                self.ghost_pts.append((g.x, g.y))
                gx, gy = zip(*self.ghost_pts)
                self.ghost_trail.set_data(gx, gy)
                self.ghost_dot.set_data([g.x], [g.y])
            else:
                self.ghost_trail.set_data([], [])
                self.ghost_dot.set_data([], [])

            modes = ["wave" if inf["mode"] == "wave" else "particle" for inf in infos]
            state = "relaxing" if self.relaxing else ("Schrödinger eq." if "wave" in modes else "Newton")
            if two:
                self.title.set_text(f"Two electrons: e₁ {modes[0]}, e₂ {modes[1]}  ({state})")
                self.mode_var.set("  ".join(f"e{k + 1}: {m.upper()}" for k, m in enumerate(modes)))
            else:
                self.title.set_text(("WAVE  |ψ|²" if modes[0] == "wave" else
                                     "PARTICLE  (classical point charge)") + f"  ({state})")
                self.mode_var.set("Mode: " + ("WAVE ψ" if modes[0] == "wave" else "PARTICLE ●"))

            # histories
            if self.relaxing is None and self.last_dt > 0:
                self.history.append((self.sim.t * AU_TIME_FS, obs["T"], obs["V"], obs["E"], *inside))
            if self.history:
                h = np.array(self.history)
                for line, col in ((self.l_t, 1), (self.l_v, 2), (self.l_e, 3)):
                    line.set_data(h[:, 0], h[:, col])
                for k, line in enumerate(self.l_in):
                    line.set_data(h[:, 0], h[:, 4 + k])
                orb = self._orbitals()
                for k, line in enumerate(self.level_lines):
                    level = orb[k][1] if (k in orb and not two) else np.nan
                    line.set_ydata([level, level])
                self._update_energy_limits(h)

            e = obs["E"]
            head = f"t = {self.sim.t * AU_TIME_FS:7.3f} fs   E = {e:8.4f} Ha ({e * HARTREE_EV:7.2f} eV)"
            if two:
                head += f"   electron repulsion V_ee = {obs['Vee']:.4f} Ha"
            orb = self._orbitals()
            levels = "  ".join(f"E_{nm} = {orb[k][1]:.4f}" for k, nm in ((0, "σ"), (1, "σ*"))
                               if k in orb)
            if levels:
                head += f"   one-electron levels: {levels}"
            self.status.set(head + "\n" + "   |   ".join(lines) + "\n" + self.message)
            self._refresh_canvas()

    root = root or tk.Tk()
    app = SimulatorApp(root)
    if mainloop:
        root.mainloop()
    return app


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if run_selftest() else 1)
    run_app()
