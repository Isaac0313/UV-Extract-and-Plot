"""
MO Electron Simulator
---------------------
One electron moving between two nuclei (an H2+-like molecule), simulated in
two ways that you can switch between at any moment:

  PARTICLE  the electron is a point charge pushed around by the Coulomb
            attraction of both nuclei (Newton's laws, F = -grad V).
  WAVE      the electron is a wavefunction psi(x, y, t), evolved with the
            time-dependent Schrodinger equation in exactly the same potential.

Switching:
  Become wave      particle at (r, p)  ->  Gaussian wave packet centred on r,
                   moving with mean momentum p (width = "packet width" slider).
  Measure          wave -> particle. Position is drawn from |psi(r)|^2 and
                   momentum from |psi(k)|^2, so on average the particle gets
                   the same energy the wave had.
  Relax -> sigma   imaginary-time evolution: the wave "cools" into the lowest
                   molecular orbital (bonding sigma) of the current molecule.
  Relax -> sigma*  same, but the bonding orbital is projected out, so the wave
                   settles into the next orbital (antibonding sigma*).
  Auto switch      random measure / become-wave events at a chosen rate.
  Ghost            while the electron is a wave, a classical "ghost" electron
                   starts from the same point so you can compare the motions.
  Classical cloud  time-average of where the particle has been; compare it to
                   |psi|^2 of the orbitals.

Mouse: click in the molecule plot to put the electron there; click-and-drag
to also give it a velocity (arrow = momentum). In WAVE mode a wave packet is
placed instead of a particle.

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
  * 2D: the electron moves in the plane that contains both nuclei.
  * Soft-core Coulomb potential V = -Z / sqrt(d^2 + a^2). The softening a
    removes the 1/r singularity so both the grid and Newton's equations stay
    well behaved; a -> 0 approaches the true Coulomb potential. Because of the
    2D geometry and the softening, energies differ from real 3D H2+.
  * Nuclei are fixed (Born-Oppenheimer). Reported electron energies do not
    include the nucleus-nucleus repulsion; "E_sigma + Z1*Z2/R" is shown
    separately so you can see whether the molecule is bound at this R.
  * Absorbing edges (optional) remove the part of the wave that leaves the
    box, so the norm then equals the probability that the electron is still
    bound (1 - ionization probability).
"""

import math
import sys
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Settings. Edit these to change the simulation; most can also be changed
# from inside the program with the sliders.
# ----------------------------------------------------------------------------
APP_TITLE = "MO Electron Simulator"
APP_SUBTITLE = "One electron, two nuclei: particle ⇄ wavefunction"
HEADER_COLOR = "#1f3b57"
HEADER_TEXT_COLOR = "#ffffff"

GRID_N = 256              # grid points per side for the wavefunction
BOX_SIZE = 40.0           # box side length, bohr (grid spacing = BOX_SIZE / GRID_N)
VIEW_HALF_WIDTH = 10.0    # the molecule plot shows -10..10 bohr
DT = 0.02                 # time step, a.u. (both particle and wave)
IMAG_DT = 0.05            # imaginary time step used by "Relax"
CLASSICAL_SUBSTEPS = 10   # Newton steps per DT (keeps energy conserved near the nuclei)
ABSORBER_WIDTH = 4.0      # width of the absorbing border, bohr
RELAX_STEPS_PER_FRAME = 20
RELAX_TOL = 1e-8          # hartree; relaxing stops when the energy changes less than this

DEFAULTS = dict(
    R=2.0,        # bond length, bohr
    Z1=1.0,       # charge of left nucleus
    Z2=1.0,       # charge of right nucleus
    soft=0.5,     # soft-core parameter a, bohr
    sigma=0.6,    # wave packet width, bohr
    steps=5,      # time steps per animation frame (simulation speed)
    rate=0.05,    # auto switch events per a.u. of time
)
START_POS = (0.0, 1.2)    # where the electron starts, bohr
START_MOM = (0.75, 0.0)   # starting momentum, a.u.
DRAG_TO_MOMENTUM = 0.4    # momentum per bohr of mouse drag

FRAME_MS = 30             # target time between redraws, ms
TRAIL_LEN = 300
HISTORY_LEN = 1500
DISPLAY_GAMMA = 0.5       # brightness = (density / max)^gamma, < 1 shows faint tails
CLOUD_BLUR = 0.3          # smoothing of the classical cloud picture, bohr

AU_TIME_FS = 0.02418884
HARTREE_EV = 27.211386


# ============================================================================
# Physics (no user interface code below until SimulatorApp)
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
        self.version = 0
        self.set_params(R, Z1, Z2, soft)

    def set_params(self, R, Z1, Z2, soft):
        self.R, self.Z1, self.Z2, self.soft = float(R), float(Z1), float(Z2), float(soft)
        self.nuclei = [(-self.R / 2, self.Z1), (self.R / 2, self.Z2)]
        self.V = self.potential(self.X, self.Y)
        self.version += 1

    def potential(self, x, y):
        a2 = self.soft ** 2
        v = 0.0
        for xn, z in self.nuclei:
            v = v - z / np.sqrt((x - xn) ** 2 + y ** 2 + a2)
        return v

    def force(self, x, y):
        """Coulomb force on the electron, F = -grad V (plain floats, fast)."""
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

    def grid_index(self, x, y):
        """(row, col) of the grid cell containing (x, y), or None if outside."""
        i = int(math.floor(x / self.dx + self.n / 2))
        j = int(math.floor(y / self.dx + self.n / 2))
        if 0 <= i < self.n and 0 <= j < self.n:
            return j, i
        return None


class ClassicalElectron:
    """Point electron, integrated with velocity Verlet."""

    def __init__(self, system, pos, mom):
        self.sys = system
        self.x, self.y = float(pos[0]), float(pos[1])
        self.px, self.py = float(mom[0]), float(mom[1])
        self.refresh_force()

    def refresh_force(self):
        self.fx, self.fy = self.sys.force(self.x, self.y)

    def step(self, dt, substeps=CLASSICAL_SUBSTEPS, cloud=None):
        h = dt / substeps
        force = self.sys.force
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
        t = 0.5 * (self.px ** 2 + self.py ** 2)
        v = float(self.sys.potential(self.x, self.y))
        return t, v, t + v


class QuantumElectron:
    """Wavefunction on the grid, propagated with the split-operator FFT method."""

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
    def step(self, n=1):
        psi = self.psi
        fft2, ifft2 = np.fft.fft2, np.fft.ifft2
        for _ in range(n):
            psi = self.half_v * psi
            psi = ifft2(self.kin * fft2(psi))
            psi = self.half_v * psi
            if self.absorb:
                psi = psi * self._mask
        self.psi = psi

    def imag_steps(self, psi, n, project_out=()):
        """Imaginary-time steps: every component decays as exp(-E t), so the
        lowest state (not projected out) survives. Returns the new array."""
        dA = self.sys.dA
        fft2, ifft2 = np.fft.fft2, np.fft.ifft2
        for _ in range(n):
            psi = self.half_v_im * psi
            psi = ifft2(self.kin_im * fft2(psi))
            psi = self.half_v_im * psi
            for phi in project_out:
                psi = psi - phi * (np.vdot(phi, psi) * dA)
            psi = self.normalized(psi)
        return psi

    def energy(self, psi):
        s = self.sys
        rho = np.abs(psi) ** 2
        w = np.abs(np.fft.fft2(psi)) ** 2
        return float(np.sum(w * s.T) / np.sum(w) + np.sum(rho * s.V) / np.sum(rho))

    def solve_state(self, seed, project_out=(), tol=RELAX_TOL, max_steps=8000, chunk=25):
        """Relax `seed` until its energy stops changing. Returns (psi, E)."""
        psi = self.normalized(seed)
        for phi in project_out:
            psi = psi - phi * (np.vdot(phi, psi) * self.sys.dA)
        e_old = np.inf
        e = self.energy(psi)
        done = 0
        while abs(e - e_old) > tol and done < max_steps:
            psi = self.imag_steps(psi, chunk, project_out)
            done += chunk
            e_old, e = e, self.energy(psi)
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

    # 1. Particle: total energy conserved by velocity Verlet.
    c = ClassicalElectron(s, START_POS, START_MOM)
    e0 = c.energies()[2]
    emax = 0.0
    for _ in range(20000):                      # 400 a.u. of time
        c.step(DT)
        emax = max(emax, abs(c.energies()[2] - e0))
    check("classical energy conservation", emax < 1e-3 * abs(e0),
          f"E0 = {e0:.6f} Ha, max drift = {emax:.2e} Ha over 400 a.u.")

    # 2. Wave: norm conserved without the absorber.
    q = QuantumElectron(s, absorb=False)
    q.from_particle(START_POS, START_MOM, p["sigma"])
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
    # Starting from a random wave packet must also reach the bonding orbital.
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
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:                                    # sharper text on Windows high-DPI screens
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    class SimulatorApp:
        def __init__(self, root):
            self.root = root
            root.title(APP_TITLE)
            self.rng = np.random.default_rng()
            p = DEFAULTS
            self.sys = TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
            self.quantum = QuantumElectron(self.sys)
            self.cmap = colormaps["inferno"]
            self.blur_k = np.exp(-0.5 * CLOUD_BLUR ** 2 * (self.sys.KX ** 2 + self.sys.KY ** 2))
            self.orbitals = {}              # {0: (psi, E), 1: (psi, E)} for the current molecule
            self.orbital_version = self.sys.version
            self.running = True
            self.drag_start = None
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
                     font=("TkDefaultFont", 15, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
            tk.Label(header, text=APP_SUBTITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR
                     ).pack(anchor="w", padx=12, pady=(0, 8))

            # Status bar and control panel are packed before the plot so they keep
            # their space when the window is small; the plot takes what is left.
            self.status = tk.StringVar()
            tk.Label(root, textvariable=self.status, anchor="w", justify="left",
                     font=("TkFixedFont", 9), relief="sunken", padx=6
                     ).pack(fill="x", side="bottom")

            body = tk.Frame(root)
            body.pack(fill="both", expand=True)
            panel = tk.Frame(body, padx=6, pady=4)
            panel.pack(side="right", fill="y")

            # --- figure -------------------------------------------------------
            self.fig = Figure(figsize=(9.6, 6.2), dpi=100)
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.42, wspace=0.25,
                                       left=0.075, right=0.98, top=0.93, bottom=0.08)
            self.ax = self.fig.add_subplot(gs[:, 0])
            self.ax_cut = self.fig.add_subplot(gs[0, 1])
            self.ax_e = self.fig.add_subplot(gs[1, 1])
            self.canvas = FigureCanvasTkAgg(self.fig, master=body)
            self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
            self._build_axes()
            self.canvas.mpl_connect("button_press_event", self.on_press)
            self.canvas.mpl_connect("motion_notify_event", self.on_motion)
            self.canvas.mpl_connect("button_release_event", self.on_release)

            # --- controls -----------------------------------------------------
            box = ttk.LabelFrame(panel, text="Electron")
            box.pack(fill="x", pady=4)
            self.mode_var = tk.StringVar()
            tk.Label(box, textvariable=self.mode_var, font=("TkDefaultFont", 11, "bold")
                     ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4)
            buttons = [
                ("Become wave ψ", self.become_wave),
                ("Measure (collapse)", self.measure),
                ("Relax → σ bonding", lambda: self.start_relax(0)),
                ("Relax → σ* anti", lambda: self.start_relax(1)),
            ]
            for k, (text, cmd) in enumerate(buttons):
                ttk.Button(box, text=text, command=cmd).grid(
                    row=1 + k // 2, column=k % 2, sticky="ew", padx=2, pady=2)
            self.play_btn = ttk.Button(box, text="Pause", command=self.toggle_run)
            self.play_btn.grid(row=3, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(box, text="Reset", command=self.reset).grid(
                row=3, column=1, sticky="ew", padx=2, pady=2)

            self.vars = {}
            box = ttk.LabelFrame(panel, text="Molecule")
            box.pack(fill="x", pady=4)
            self._slider(box, "R", "Bond length R (bohr)", 0.4, 8.0, 0.05, self.on_molecule)
            self._slider(box, "Z1", "Left nucleus charge Z₁", 0.5, 3.0, 0.05, self.on_molecule)
            self._slider(box, "Z2", "Right nucleus charge Z₂", 0.5, 3.0, 0.05, self.on_molecule)
            self._slider(box, "soft", "Soft-core a (bohr)", 0.2, 1.5, 0.05, self.on_molecule)

            box = ttk.LabelFrame(panel, text="Simulation")
            box.pack(fill="x", pady=4)
            self._slider(box, "sigma", "Wave packet width (bohr)", 0.2, 2.0, 0.05, None)
            self._slider(box, "steps", "Speed (steps / frame)", 1, 25, 1, None)
            self._slider(box, "rate", "Auto switch rate (1/a.u.)", 0.005, 0.5, 0.005, None)

            box = ttk.LabelFrame(panel, text="Options")
            box.pack(fill="x", pady=4)
            self.flags = {}
            for key, text, val in [("auto", "Auto switch (random)", False),
                                   ("ghost", "Classical ghost while wave", True),
                                   ("cloud", "Classical cloud (time average)", False),
                                   ("absorb", "Absorbing edges (ionization)", True),
                                   ("phase", "Show phase as colour", False)]:
                v = tk.BooleanVar(value=val)
                ttk.Checkbutton(box, text=text, variable=v, command=self.on_flag).pack(anchor="w")
                self.flags[key] = v

            tk.Label(panel, justify="left", fg="#555", wraplength=240,
                     text="Click the plot to place the electron;\n"
                          "drag to give it a velocity.").pack(anchor="w", pady=(4, 0))

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
                     ).pack(fill="x", padx=4, pady=(0, 3))

        def _build_axes(self):
            s = self.sys
            ax = self.ax
            # Only the visible part of the grid is turned into an image (faster).
            i0 = int(np.searchsorted(s.x, -VIEW_HALF_WIDTH - 2 * s.dx))
            self.view = slice(i0, s.n - i0)
            h = s.dx / 2
            xv = s.x[self.view]
            ext = (xv[0] - h, xv[-1] + h, xv[0] - h, xv[-1] + h)
            self.img = ax.imshow(np.zeros((xv.size, xv.size, 3), np.uint8), origin="lower", extent=ext,
                                 interpolation="bilinear")
            ax.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            ax.set_ylim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            ax.set_xlabel("x (bohr)")
            ax.set_ylabel("y (bohr)")
            self.title = ax.set_title("")
            self.contours = None
            self.nuclei_art = ax.scatter([], [], s=[], c="#e8483b", edgecolors="white",
                                         linewidths=1.2, zorder=5)
            (self.trail,) = ax.plot([], [], "-", color="#7fdcff", lw=1.2, alpha=0.8, zorder=6)
            (self.dot,) = ax.plot([], [], "o", color="#7fdcff", ms=9, mec="white", zorder=7)
            (self.ghost_trail,) = ax.plot([], [], "--", color="white", lw=0.9, alpha=0.6, zorder=6)
            (self.ghost_dot,) = ax.plot([], [], "o", mfc="none", mec="white", ms=9, zorder=7)
            (self.drag_line,) = ax.plot([], [], "-", color="#ffe066", lw=2, zorder=8)

            a = self.ax_cut
            a.set_xlabel("x along the bond, cut at y = 0 (bohr)")
            a.set_ylabel("energy (hartree)")
            (self.cut_v,) = a.plot(s.x, s.V[s.n // 2], color="#555", lw=1.5, label="V(x, 0)")
            (self.cut_rho,) = a.plot([], [], color="#d9480f", lw=1.5, label="|ψ(x,0)|² (scaled)")
            (self.cut_e,) = a.plot([], [], ":", color="#1c7ed6", lw=1.2, label="electron energy")
            (self.cut_dot,) = a.plot([], [], "o", color="#1c7ed6", ms=7)
            a.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_e
            a.set_xlabel("time (fs)")
            a.set_ylabel("hartree")
            (self.l_t,) = a.plot([], [], color="#2f9e44", lw=1.2, label="kinetic T")
            (self.l_v,) = a.plot([], [], color="#7048e8", lw=1.2, label="potential V")
            (self.l_e,) = a.plot([], [], color="black", lw=1.6, label="total E")
            self.level_lines = [a.axhline(np.nan, color=c, ls="--", lw=0.9)
                                for c in ("#1c7ed6", "#e8590c")]
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            # Blitting: everything that changes every frame is "animated" and redrawn
            # on top of a cached picture of the static parts (axes, ticks, labels).
            self.anim = [self.img, self.nuclei_art, self.trail, self.ghost_trail, self.dot,
                         self.ghost_dot, self.drag_line, self.title,
                         self.cut_rho, self.cut_e, self.cut_dot,
                         self.l_t, self.l_v, self.l_e, *self.level_lines]
            for art in self.anim:
                art.set_animated(True)
            self.background = None
            self.need_full_draw = True
            self.canvas.mpl_connect("draw_event", self._on_draw)

        def _on_draw(self, _event):
            self.background = self.canvas.copy_from_bbox(self.fig.bbox)
            self._draw_animated()

        def _draw_animated(self):
            for art in self.anim:
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
            vals = h[:, 1:]
            lo, hi = float(vals.min()), float(vals.max())
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
                                            linewidths=0.5, alpha=0.35, zorder=3)
            self.nuclei_art.set_offsets([(xn, 0) for xn, _ in s.nuclei])
            self.nuclei_art.set_sizes([60 + 60 * z for _, z in s.nuclei])
            self.cut_v.set_ydata(s.V[s.n // 2])
            self.ax_cut.set_ylim(s.V.min() - 0.3, 1.0)
            self.need_full_draw = True

        # ------------------------------------------------------------------
        # State changes
        # ------------------------------------------------------------------
        def reset(self):
            p = DEFAULTS
            for k, v in self.vars.items():
                v.set(p[k])
            self.sys.set_params(p["R"], p["Z1"], p["Z2"], p["soft"])
            self.quantum.rebuild()
            self.quantum.absorb = self.flags["absorb"].get()
            self.t = 0.0
            self.history = deque(maxlen=HISTORY_LEN)
            self.cloud = np.zeros((self.sys.n, self.sys.n))
            self.message = "Particle mode: the electron is a point charge pulled by both nuclei."
            self.place(START_POS, START_MOM, as_wave=False)
            self._draw_molecule()

        def place(self, pos, mom, as_wave):
            self.relaxing = None
            self.ghost = None
            self.trail_pts = deque(maxlen=TRAIL_LEN)
            self.ghost_pts = deque(maxlen=TRAIL_LEN)
            if as_wave:
                self.mode = "wave"
                self.quantum.from_particle(pos, mom, self.vars["sigma"].get())
                if self.flags["ghost"].get():
                    self.ghost = ClassicalElectron(self.sys, pos, mom)
            else:
                self.mode = "particle"
                self.particle = ClassicalElectron(self.sys, pos, mom)

        def become_wave(self):
            if self.mode == "wave":
                self.message = "Already a wave. Use Measure to turn it back into a particle."
                return
            c = self.particle
            e_particle = c.energies()[2]
            self.place((c.x, c.y), (c.px, c.py), as_wave=True)
            e_wave = self.quantum.energy(self.quantum.psi)
            self.message = (f"Became a wave packet (same <r>, <p>). Energy {e_particle:.3f} → "
                            f"{e_wave:.3f} Ha: squeezing it into width σ costs kinetic energy "
                            f"(uncertainty principle). Watch it spread and tunnel.")

        def measure(self):
            if self.mode != "wave":
                self.message = "Already a particle. Use Become wave first."
                return
            pos, mom = self.quantum.measure(self.rng)
            self.place(pos, mom, as_wave=False)
            e = self.particle.energies()[2]
            self.message = (f"Measured! Electron found at ({pos[0]:.2f}, {pos[1]:.2f}) bohr, "
                            f"energy {e:.3f} Ha" + ("  (unbound: E > 0, it will escape)" if e > 0 else ""))

        def start_relax(self, target):
            if self.mode == "particle":
                c = self.particle
                self.place((c.x, c.y), (c.px, c.py), as_wave=True)
            q = self.quantum
            project = []
            if target == 1:
                if 0 not in self._orbitals():
                    self.status.set("Finding the bonding orbital first ...")
                    self.root.update_idletasks()
                    self.orbitals[0] = q.solve_state(q.lcao(+1))
                project = [self.orbitals[0][0]]
            # A little LCAO guess makes sure the target orbital is present in the start state.
            psi = q.normalized(q.psi) + 0.05 * q.lcao(+1 if target == 0 else -1)
            for phi in project:
                psi = psi - phi * (np.vdot(phi, psi) * self.sys.dA)
            q.psi = q.normalized(psi)
            self.relaxing = (target, project, np.inf)
            self.ghost = None
            self.ghost_pts.clear()
            name = "σ bonding" if target == 0 else "σ* antibonding"
            self.message = f"Relaxing (imaginary time) into the {name} orbital ..."

        def _orbitals(self):
            if self.orbital_version != self.sys.version:
                self.orbitals = {}
                self.orbital_version = self.sys.version
            return self.orbitals

        def on_molecule(self):
            v = self.vars
            self.sys.set_params(v["R"].get(), v["Z1"].get(), v["Z2"].get(), v["soft"].get())
            self.quantum.rebuild()
            for c in (getattr(self, "particle", None), self.ghost):
                if c is not None:
                    c.refresh_force()
            self.cloud[:] = 0
            self._draw_molecule()

        def on_flag(self):
            self.quantum.absorb = self.flags["absorb"].get()
            if not self.flags["cloud"].get():
                self.cloud[:] = 0
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
            self.place((x0, y0), mom, as_wave=(self.mode == "wave"))
            self.message = (f"Placed {'wave packet' if self.mode == 'wave' else 'electron'} at "
                            f"({x0:.2f}, {y0:.2f}) with momentum ({mom[0]:.2f}, {mom[1]:.2f}).")
            self.render()

        # ------------------------------------------------------------------
        # Animation loop
        # ------------------------------------------------------------------
        def tick(self):
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
            steps = int(self.vars["steps"].get())
            cloud = self.cloud if self.flags["cloud"].get() else None
            q = self.quantum

            if self.relaxing is not None:
                target, project, e_old = self.relaxing
                q.psi = q.imag_steps(q.psi, RELAX_STEPS_PER_FRAME, project)
                e = q.energy(q.psi)
                if abs(e - e_old) < RELAX_TOL:
                    self._orbitals()[target] = (q.psi.copy(), e)
                    self.relaxing = None
                    name = "σ bonding" if target == 0 else "σ* antibonding"
                    extra = ""
                    if target == 0:
                        tot = e + self.sys.nuclear_repulsion()
                        extra = f"; with nuclear repulsion {tot:.4f} Ha ({'bound' if tot < 0 else 'not bound'})"
                    self.message = (f"Settled into the {name} orbital: E = {e:.4f} Ha = "
                                    f"{e * HARTREE_EV:.2f} eV{extra}. It is now a stationary state.")
                else:
                    self.relaxing = (target, project, e)
                return

            if self.mode == "particle":
                for _ in range(steps):
                    self.particle.step(DT, cloud=cloud)
            else:
                q.step(steps)
                if self.ghost is not None:
                    for _ in range(steps):
                        self.ghost.step(DT, cloud=cloud)
            self.t += steps * DT

            if self.flags["auto"].get():
                if self.rng.random() < 1 - math.exp(-self.vars["rate"].get() * steps * DT):
                    if self.mode == "particle":
                        self.become_wave()
                    else:
                        self.measure()

        # ------------------------------------------------------------------
        # Drawing
        # ------------------------------------------------------------------
        def _image(self, rho, psi=None):
            v = self.view
            rho = rho[v, v]
            psi = None if psi is None else psi[v, v]
            m = rho.max()
            b = (rho / m) ** DISPLAY_GAMMA if m > 0 else rho
            if psi is not None and self.flags["phase"].get():
                hsv = np.stack([(np.angle(psi) + np.pi) / (2 * np.pi), np.ones_like(b), b], axis=-1)
                return (hsv_to_rgb(hsv) * 255).astype(np.uint8)
            return self.cmap(b, bytes=True)[..., :3]

        def render(self):
            s = self.sys
            cut_rho = None
            if self.mode == "particle":
                c = self.particle
                t, v, e = c.energies()
                x, y = c.x, c.y
                if self.flags["cloud"].get() and self.cloud.max() > 0:
                    # Smooth the visit histogram a little (Gaussian blur done with FFTs).
                    cloud = np.fft.ifft2(np.fft.fft2(self.cloud) * self.blur_k).real.clip(0)
                    self.img.set_data(self._image(cloud))
                    cut_rho = cloud[s.n // 2]
                else:
                    self.img.set_data(np.zeros(self.img.get_array().shape))
                self.dot.set_data([x], [y])
                self.dot.set_marker("o")
                self.cut_dot.set_data([x], [e])
                info = f"r = ({x:6.2f}, {y:6.2f})  p = ({c.px:5.2f}, {c.py:5.2f})"
                title = "PARTICLE  (classical point charge)"
            else:
                q = self.quantum
                ex = q.expectations()
                t, v, e = ex["T"], ex["V"], ex["E"]
                x, y = ex["x"], ex["y"]
                rho = np.abs(q.psi) ** 2
                self.img.set_data(self._image(rho, q.psi))
                cut_rho = rho[s.n // 2]
                self.dot.set_data([x], [y])
                self.dot.set_marker("+")
                self.cut_dot.set_data([], [])
                info = (f"<r> = ({x:6.2f}, {y:6.2f})  <p> = ({ex['px']:5.2f}, {ex['py']:5.2f})  "
                        f"bound probability = {ex['norm']:.3f}")
                title = "WAVE  |ψ|²  (" + ("relaxing" if self.relaxing else
                                                    "Schrödinger equation") + ")"
            self.title.set_text(title)
            self.mode_var.set("Mode: " + ("PARTICLE ●" if self.mode == "particle" else "WAVE ψ"))

            self.trail_pts.append((x, y))
            tx, ty = zip(*self.trail_pts)
            self.trail.set_data(tx, ty)
            if self.ghost is not None:
                g = self.ghost
                self.ghost_pts.append((g.x, g.y))
                gx, gy = zip(*self.ghost_pts)
                self.ghost_trail.set_data(gx, gy)
                self.ghost_dot.set_data([g.x], [g.y])
            else:
                self.ghost_trail.set_data([], [])
                self.ghost_dot.set_data([], [])

            # cut along the bond
            self.cut_e.set_data([s.x[0], s.x[-1]], [e, e])
            if cut_rho is not None and cut_rho.max() > 0:
                self.cut_rho.set_data(s.x, e + 0.8 * cut_rho / cut_rho.max())
            else:
                self.cut_rho.set_data([], [])

            # energy history
            if self.relaxing is None:
                self.history.append((self.t * AU_TIME_FS, t, v, e))
            if self.history:
                h = np.array(self.history)
                for line, col in ((self.l_t, 1), (self.l_v, 2), (self.l_e, 3)):
                    line.set_data(h[:, 0], h[:, col])
                orb = self._orbitals()
                for k, line in enumerate(self.level_lines):
                    level = orb[k][1] if k in orb else np.nan
                    line.set_ydata([level, level])
                self._update_energy_limits(h)

            orb = self._orbitals()
            levels = "  ".join(f"E_{n} = {orb[k][1]:.4f}" for k, n in ((0, "σ"), (1, "σ*")) if k in orb)
            self.status.set(f"t = {self.t * AU_TIME_FS:8.3f} fs   E = {e:8.4f} Ha ({e * HARTREE_EV:7.2f} eV)   "
                            f"{info}   {levels}\n{self.message}")
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
