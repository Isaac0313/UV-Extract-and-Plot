"""
mo_controller.py - what the MO Electron Simulator *does*, without any window
---------------------------------------------------------------------------
The windows (tkinter + matplotlib in MO-Electron-Simulator.py, pyqtgraph in
MO-Electron-Simulator-Qt.py) only draw and forward clicks. Everything else -
buttons, sliders, options, stepping the simulation, tracking whether the
electrons stay inside the orbital, histories and the status text - lives
here, so both windows behave the same and either physics backend
(mo_physics or mo_physics_jax) can be plugged in.

Typical use by a window, once per frame:
    ctl.housekeeping(dragging)     # deferred work (e.g. new sigma orbital after a slider moved)
    ctl.advance()                  # run the simulation
    frame = ctl.frame()            # everything needed to draw this frame
"""

import math
import time
from collections import deque

import numpy as np

import mo_physics

# ----------------------------------------------------------------------------
# Settings for the windows (the physics settings are in mo_physics.py)
# ----------------------------------------------------------------------------
VIEW_HALF_WIDTH = 10.0    # the molecule plot shows -10..10 bohr
RELAX_STEPS_PER_FRAME = 20          # one electron
RELAX_STEPS_PER_FRAME_2E = 8        # two electrons (each step is more work)
RELAX_STEPS_PER_FRAME_SHARED = 15   # two electrons sharing one orbital
SIGMA_TRACK_STEPS = 3     # per frame, keeps the sigma orbital matched to moving nuclei
DRAG_TO_MOMENTUM = 0.4    # momentum per bohr of mouse drag
ELECTRON_COLORS = ("#4dabf7", "#ff922b")       # electrons on the dark molecule plot
ELECTRON_LINE_COLORS = ("#1971c2", "#e8590c")  # electrons on the white plots
FRAME_MS = 30             # target time between redraws, ms
TRAIL_LEN = 300
HISTORY_LEN = 1500
DISPLAY_GAMMA = 0.5       # brightness = (density / max)^gamma, < 1 shows faint tails
CLOUD_BLUR = 0.3          # smoothing of the classical cloud picture, bohr

FLAG_DEFAULTS = dict(auto=False, ghost=True, cloud=False, absorb=True, phase=False, vibrate=False)
FLAG_LABELS = [("auto", "Auto switch (random)"),
               ("ghost", "Classical ghost (1 electron)"),
               ("cloud", "Classical cloud (time average)"),
               ("absorb", "Absorbing edges (ionization)"),
               ("phase", "Show phase (1 electron)"),
               ("vibrate", "Nuclei vibrate (bond can stretch)")]
# (key, label, min, max, step) for the sliders; "molecule" ones rebuild the molecule
MOLECULE_SLIDERS = [("R", "Bond length R (bohr)", 0.4, 8.0, 0.05),
                    ("Z1", "Left nucleus charge Z₁", 0.5, 3.0, 0.05),
                    ("Z2", "Right nucleus charge Z₂", 0.5, 3.0, 0.05),
                    ("soft", "Soft-core a (bohr)", 0.2, 1.5, 0.05),
                    ("mass", "Nuclear mass (H = 1836)", 20, 3700, 1)]
SIMULATION_SLIDERS = [("sigma", "Wave packet width (bohr)", 0.2, 2.0, 0.05),
                      ("steps", "Speed (steps / frame)", 1, 25, 1),
                      ("rate", "Auto switch rate (1/a.u.)", 0.005, 0.5, 0.005)]


def hex_rgb(color):
    color = color.lstrip("#")
    return np.array([int(color[k:k + 2], 16) / 255 for k in (0, 2, 4)])


def hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB (all inputs in 0..1); returns (..., 3)."""
    i = np.floor(h * 6) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    choices = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    rgb = [np.select([i == k for k in range(6)], [c[ch] for c in choices]) for ch in range(3)]
    return np.stack(rgb, axis=-1)


def default_lut():
    """256 x 3 uint8 colour table ("inferno") used for single-electron images."""
    try:
        from matplotlib import colormaps
        return (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:
        try:
            import pyqtgraph as pg
            return pg.colormap.get("inferno").getLookupTable(nPts=256)[:, :3].astype(np.uint8)
        except Exception:
            g = np.linspace(0, 255, 256).astype(np.uint8)
            return np.stack([g, g, g], axis=1)


class Controller:
    def __init__(self, physics=mo_physics, lut=None, busy=None):
        """physics: mo_physics or mo_physics_jax. busy(text): optional callback the
        window uses to show a message before a slow (blocking) computation."""
        self.m = physics
        self.busy = busy or (lambda text: None)
        self.lut = default_lut() if lut is None else lut
        self.electron_rgb = [hex_rgb(c) for c in ELECTRON_COLORS]
        self.params = dict(physics.DEFAULTS)
        self.flags = dict(FLAG_DEFAULTS)
        self.target = "1"
        self.rng = np.random.default_rng()
        p = self.params
        self.sys = physics.TwoCenterSystem(p["R"], p["Z1"], p["Z2"], p["soft"])
        self.sim = physics.Simulation(self.sys)
        s = self.sys
        self.blur_k = np.exp(-0.5 * CLOUD_BLUR ** 2 * (s.KX ** 2 + s.KY ** 2))
        i0 = int(np.searchsorted(s.x, -VIEW_HALF_WIDTH - 2 * s.dx))
        self.view = slice(i0, s.n - i0)              # visible part of the grid
        h = s.dx / 2
        xv = s.x[self.view]
        self.view_x = xv
        self.extent = (xv[0] - h, xv[-1] + h, xv[0] - h, xv[-1] + h)
        self.orbitals = {}              # {0: (psi, E), 1: (psi, E)} one-electron orbitals
        self.orbital_version = -1
        self.region = None              # (mask, threshold) of the sigma orbital
        self.region_version = 0         # bumps when the 90 % boundary changes -> redraw it
        self.molecule_version = 0       # bumps when the potential changes -> redraw contours
        self.molecule_rescale = True    # the cut plot's y range should be reset
        self.region_dirty_since = None
        self.trails = [deque(maxlen=TRAIL_LEN) for _ in range(2)]
        self.ghost_pts = deque(maxlen=TRAIL_LEN)
        self.inside_acc = [[0.0, 0.0], [0.0, 0.0]]   # [time inside, total time] per electron
        self.last_dt = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @property
    def count(self):
        return self.sim.count

    def reset(self):
        self.params = dict(self.m.DEFAULTS)
        p = self.params
        self.sys.set_params(p["R"], p["Z1"], p["Z2"], p["soft"])
        self.sim.rebuild()
        self.sim.set_absorb(self.flags["absorb"])
        self.sim.nuclear_mass = float(p["mass"])
        self.sim.nuc_p = [0.0, 0.0]
        self.sim.t = 0.0
        self.history = deque(maxlen=HISTORY_LEN)
        self.clouds = [np.zeros((self.sys.n, self.sys.n)) for _ in range(2)]
        self.relaxing = None
        self.ghost = None
        self.ghost_pts.clear()
        for i in range(2):
            self.place(i, *self.m.START[i], as_wave=False)
        self._molecule_changed(rescale=True)
        self.ensure_sigma()
        self.message = ("Particle mode: point charges pulled by the nuclei. Click the plot to place "
                        "an electron, drag to give it a velocity.")

    def set_param(self, key, value):
        self.params[key] = float(value)
        if key in ("R", "Z1", "Z2", "soft"):
            self.apply_molecule()
        elif key == "mass":
            self.sim.nuclear_mass = float(value)

    def apply_molecule(self):
        p = self.params
        self.sys.set_params(p["R"], p["Z1"], p["Z2"], p["soft"])
        self.sim.rebuild()
        self.sim.nuc_p = [0.0, 0.0]          # nuclei restart at rest at the new R
        if self.ghost is not None:
            self.ghost.refresh_force()
        for c in self.clouds:
            c[:] = 0
        self._orbitals()                    # drops the old sigma and its boundary
        self.region_dirty_since = time.time()
        self._molecule_changed(rescale=True)

    def set_flag(self, key, value):
        value = bool(value)
        self.flags[key] = value
        if key == "vibrate" and value != self.sim.nuclei_move:
            self.sim.nuclei_move = value
            self.sim.nuc_p = [0.0, 0.0]
            self.sim.nuclear_mass = float(self.params["mass"])
            self.history.clear()              # the energy now does / does not include the nuclei
            if value:
                self._orbitals().pop(1, None)
                self.message = ("Nuclei now move along the bond. The electrons pull them together and they "
                                "repel each other; start away from the balance point (R slider) to see a "
                                "vibration. With the real H mass one vibration takes ~10 fs (thousands of "
                                "frames): lower the nuclear mass or raise the speed to watch it sooner.")
            else:
                self.params["R"] = round(self.sys.R, 2)
                self.message = f"Nuclei fixed again at R = {self.sys.R:.3f} bohr."
        if key == "absorb":
            self.sim.set_absorb(value)
        if key == "cloud" and not value:
            for c in self.clouds:
                c[:] = 0
        if key == "ghost" and not value:
            self.ghost = None
            self.ghost_pts.clear()

    def set_target(self, target):
        self.target = target                # "1", "2" or "both"

    def set_count(self, n):
        n = int(n)
        if n == self.sim.count:
            return
        self.sim.count = n
        self.relaxing = None
        if n == 2:
            # Start the new electron at the mirror image of electron 1 (r -> -r, p -> -p).
            e1, e2 = self.sim.electrons
            if e1.mode == "particle":
                c = e1.particle
                xc = (self.sys.x1 + self.sys.x2) / 2     # centre of the molecule
                self.place(1, (2 * xc - c.x, -c.y), (-c.px, -c.py), as_wave=False)
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

    # ------------------------------------------------------------------
    # Orbitals and the "inside sigma" region
    # ------------------------------------------------------------------
    def _orbitals(self):
        if self.orbital_version != self.sys.params_version:
            self.orbitals = {}
            self.region = None
            self.region_version += 1
            self.orbital_version = self.sys.params_version
        return self.orbitals

    def ensure_sigma(self):
        """One-electron sigma orbital of the current molecule and its 90 % region."""
        orb = self._orbitals()
        if 0 not in orb:
            self.busy("Computing the σ orbital of this molecule ...")
            q = self.sim.electrons[0].wave
            orb[0] = q.solve_state(q.lcao(+1))
        if self.region is None:
            self._set_region(orb[0][0])
        return orb[0]

    def _set_region(self, psi):
        self.region = self.m.orbital_region(psi, self.sys.dA)
        self.region_version += 1

    def sigma_density(self):
        """|sigma|^2 on the full grid (for drawing the 90 % boundary), or None."""
        orb = self._orbitals()
        return None if 0 not in orb else np.abs(np.asarray(orb[0][0])) ** 2

    def _molecule_changed(self, rescale):
        self.molecule_version += 1
        self.molecule_rescale = self.molecule_rescale or rescale
        self.drawn_geom = (self.sys.x1, self.sys.x2)
        if rescale:
            self.cut_ylim = (float(self.sys.V.min()) - 0.3, 1.0)

    def _follow_nuclei(self):
        """While the nuclei move: keep sigma matched to them (a few imaginary-time
        steps per frame) and redraw potential and boundary when they moved noticeably."""
        s = self.sys
        orb = self._orbitals()
        orb.pop(1, None)                       # sigma* is not tracked while vibrating
        q = self.sim.electrons[0].wave
        if 0 in orb:
            orb[0] = (q.imag_steps(orb[0][0], SIGMA_TRACK_STEPS), orb[0][1])
        moved = max(abs(s.x1 - self.drawn_geom[0]), abs(s.x2 - self.drawn_geom[1]))
        if moved > 0.03:
            if 0 in orb:
                orb[0] = (orb[0][0], q.energy(orb[0][0]))
                self._set_region(orb[0][0])
            self._molecule_changed(rescale=False)

    def housekeeping(self, dragging=False):
        """Deferred work: a new sigma orbital ~0.4 s after the molecule sliders stop."""
        if self.region_dirty_since and time.time() - self.region_dirty_since > 0.4 and not dragging:
            self.region_dirty_since = None
            self.ensure_sigma()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def targets(self):
        if self.sim.count == 1:
            return [0]
        return {"1": [0], "2": [1], "both": [0, 1]}[self.target]

    def place(self, i, pos, mom, as_wave):
        e = self.sim.electrons[i]
        if as_wave:
            e.set_wave(pos, mom, self.params["sigma"])
        else:
            e.set_particle(pos, mom)
        self.trails[i].clear()
        self.inside_acc[i] = [0.0, 0.0]
        self.relaxing = None
        if i == 0:
            self.ghost = None
            self.ghost_pts.clear()
            if as_wave and self.sim.count == 1 and self.flags["ghost"]:
                self.ghost = self.m.ClassicalElectron(self.sys, pos, mom)

    def place_from_drag(self, x0, y0, x1, y1):
        """Mouse: press at (x0, y0), release at (x1, y1)."""
        mom = (DRAG_TO_MOMENTUM * (x1 - x0), DRAG_TO_MOMENTUM * (y1 - y0))
        i = self.targets()[0]
        as_wave = self.sim.electrons[i].mode == "wave"
        self.place(i, (x0, y0), mom, as_wave=as_wave)
        label = f"electron {i + 1}" if self.sim.count == 2 else "electron"
        self.message = (f"Placed {label} ({'wave packet' if as_wave else 'particle'}) at "
                        f"({x0:.2f}, {y0:.2f}) with momentum ({mom[0]:.2f}, {mom[1]:.2f}).")

    def _to_wave(self, i, for_relax=False):
        """Particle -> wave packet. An electron that has escaped the molecule is
        refused (a packet at the box edge would wrap around), unless we are
        about to relax it anyway, in which case it restarts at the centre."""
        c = self.sim.electrons[i].particle
        escape = self.sys.box / 2 - self.m.ABSORBER_WIDTH - 2.0
        if max(abs(c.x), abs(c.y)) > escape:
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
        dA = self.sys.dA
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
                    g = self.ensure_sigma()[0]
                else:
                    # lowest orbital in the field of the (frozen) other electron
                    self.busy("Finding the lowest orbital first ...")
                    g = q.solve_state(q.lcao(+1), v_extra=self.sim.field_of(other))[0]
                project[i] = [g]
            # A little LCAO guess makes sure the target orbital is present in the start state.
            psi = q.normalized(q.psi) + 0.05 * q.lcao(+1 if target == 0 else -1)
            for phi in project.get(i, ()):
                psi = psi - phi * (np.vdot(np.asarray(phi), np.asarray(psi)) * dA)
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
                            f"{e_total * self.m.HARTREE_EV:.2f} eV{extra}. It is now a stationary state.")
        else:
            who = "Both electrons" if len(r["idxs"]) == 2 else f"Electron {r['idxs'][0] + 1}"
            where = "one shared " if r["shared"] else "the "
            self.message = (f"{who} settled into {where}{name} orbital (mean field){note}. "
                            f"Total electronic E = {e_total:.4f} Ha, + nuclear repulsion = "
                            f"{e_total + rep:.4f} Ha.")

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def advance(self):
        self.last_dt = 0.0
        m = self.m
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
            if abs(e - r["e_old"]) < m.RELAX_TOL or r["steps"] >= m.RELAX_MAX_STEPS:
                self._finish_relax(e, abs(e - r["e_old"]) < m.RELAX_TOL)
            else:
                r["e_old"] = e
            return

        steps = int(self.params["steps"])
        clouds = self.clouds if self.flags["cloud"] else (None, None)
        self.sim.advance(steps, clouds)
        if self.ghost is not None:
            for _ in range(steps):
                self.ghost.step(m.DT)
        self.last_dt = steps * m.DT
        if self.sim.nuclei_move:
            self._follow_nuclei()

        if self.flags["auto"]:
            if self.rng.random() < 1 - math.exp(-self.params["rate"] * steps * m.DT):
                i = int(self.rng.integers(self.sim.count))
                if self.sim.electrons[i].mode == "particle":
                    if self._to_wave(i):
                        self.message = f"Auto: electron {i + 1} became a wave."
                else:
                    pos, mom = self.sim.electrons[i].wave.measure(self.rng)
                    self.place(i, pos, mom, as_wave=False)
                    self.message = f"Auto: electron {i + 1} measured at ({pos[0]:.2f}, {pos[1]:.2f})."

    # ------------------------------------------------------------------
    # What to draw
    # ------------------------------------------------------------------
    def _brightness(self, rho):
        rho = rho[self.view, self.view]
        mx = rho.max()
        return (rho / mx) ** DISPLAY_GAMMA if mx > 0 else rho

    def _smoothed_cloud(self, i):
        c = self.clouds[i]
        if not self.flags["cloud"] or c.max() == 0:
            return None
        return np.fft.ifft2(np.fft.fft2(c) * self.blur_k).real.clip(0)

    def _make_image(self, infos):
        shape = (self.view_x.size, self.view_x.size, 3)
        if len(infos) == 1:
            info = infos[0]
            layer = info["rho"] if info["mode"] == "wave" else self._smoothed_cloud(0)
            if layer is None:
                return np.zeros(shape, np.uint8)
            b = self._brightness(layer)
            if info["mode"] == "wave" and self.flags["phase"]:
                psi = np.asarray(self.sim.electrons[0].wave.psi)[self.view, self.view]
                rgb = hsv_to_rgb((np.angle(psi) + np.pi) / (2 * np.pi), np.ones_like(b), b)
                return (rgb * 255).astype(np.uint8)
            return self.lut[(b * 255).astype(np.int32)]
        rgb = np.zeros(shape)
        for i, info in enumerate(infos):     # one colour per electron, added together
            layer = info["rho"] if info["mode"] == "wave" else self._smoothed_cloud(i)
            if layer is not None:
                rgb += self._brightness(layer)[..., None] * self.electron_rgb[i]
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _inside(self, info):
        if self.region is None:
            return np.nan
        mask = self.region[0]
        if info["mode"] == "particle":
            idx = self.sys.grid_index(info["x"], info["y"])
            return float(mask[idx]) if idx is not None else 0.0
        return float(np.sum(info["rho"] * mask) * self.sys.dA)

    def frame(self):
        """Everything a window needs to draw this frame (also updates trails/histories)."""
        s, m = self.sys, self.m
        obs = self.sim.observe()
        for info in obs["electrons"]:
            if "rho" in info:
                info["rho"] = np.asarray(info["rho"])
        infos = obs["electrons"]
        two = len(infos) == 2
        orb = self._orbitals()
        sigma = orb.get(0)
        out = dict(two=two, image=self._make_image(infos), extent=self.extent,
                   nuclei=[(xn, z) for xn, z in s.nuclei], electrons=[], ghost=None,
                   cut_x=s.x, cut_v=None, cut_lines=[None, None], cut_dots=[None, None], cut_energy=None,
                   cut_ylim=self.cut_ylim)
        if self.molecule_version != getattr(self, "_cut_v_version", None) or self.sim.nuclei_move:
            out["cut_v"] = np.asarray(s.V[s.n // 2])
            self._cut_v_version = self.molecule_version

        inside, lines = [np.nan, np.nan], []
        for i, info in enumerate(infos):
            x, y = info["x"], info["y"]
            is_wave = info["mode"] == "wave"
            self.trails[i].append((x, y))
            tx, ty = (np.array(v) for v in zip(*self.trails[i]))
            out["electrons"].append(dict(x=x, y=y, wave=is_wave, trail=(tx, ty)))

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
                    psi_i = np.asarray(self.sim.electrons[i].wave.psi)
                    p_sigma = abs(np.vdot(np.asarray(sigma[0]), psi_i)) ** 2 * s.dA ** 2
                    pop = f"  P(σ) = {p_sigma:.2f}"
                lines.append(f"{label}WAVE <r> = ({x:5.2f}, {y:5.2f})  in σ region {ins:4.0%}{avg}"
                             f"{pop}  bound {info['norm']:.3f}")
            else:
                lines.append(f"{label}PARTICLE r = ({x:5.2f}, {y:5.2f})  p = ({info['px']:5.2f}, "
                             f"{info['py']:5.2f})  in σ region: {'yes' if ins >= 0.5 else 'no '}{avg}")

            # cut along the bond (y = 0)
            if two:
                base = self.cut_ylim[0] + 0.1
                row = info["rho"][s.n // 2] if is_wave else None
                if row is not None and row.max() > 0:
                    out["cut_lines"][i] = base + 1.2 * row / row.max()
                out["cut_dots"][i] = None if is_wave else (x, base)
            else:
                e1 = info["T"] + info["V"]
                row = info["rho"][s.n // 2] if is_wave else None
                if row is None:
                    cl = self._smoothed_cloud(0)
                    row = None if cl is None else cl[s.n // 2]
                if row is not None and row.max() > 0:
                    out["cut_lines"][i] = e1 + 0.8 * row / row.max()
                out["cut_dots"][i] = None if is_wave else (x, e1)
                out["cut_energy"] = e1

        if self.ghost is not None and not two:
            g = self.ghost
            self.ghost_pts.append((g.x, g.y))
            gx, gy = (np.array(v) for v in zip(*self.ghost_pts))
            out["ghost"] = dict(x=g.x, y=g.y, trail=(gx, gy))

        modes = [inf["mode"] for inf in infos]
        state = "relaxing" if self.relaxing else ("Schrödinger eq." if "wave" in modes else "Newton")
        if self.sim.nuclei_move and not self.relaxing:
            state += ", nuclei vibrating"
        if two:
            out["title"] = f"Two electrons: e₁ {modes[0]}, e₂ {modes[1]}  ({state})"
            out["mode_label"] = "  ".join(f"e{k + 1}: {mm.upper()}" for k, mm in enumerate(modes))
        else:
            out["title"] = (("WAVE  |ψ|²" if modes[0] == "wave" else
                             "PARTICLE  (classical point charge)") + f"  ({state})")
            out["mode_label"] = "Mode: " + ("WAVE ψ" if modes[0] == "wave" else "PARTICLE ●")

        # histories: time (fs), T, V, E, inside e1, inside e2, R
        if self.relaxing is None and self.last_dt > 0:
            self.history.append((self.sim.t * m.AU_TIME_FS, obs["T"], obs["V"], obs["E"], *inside, s.R))
        out["history"] = np.array(self.history) if self.history else None
        show_levels = not two and not self.sim.nuclei_move
        out["levels"] = [orb[k][1] if (k in orb and show_levels) else np.nan for k in (0, 1)]

        e = obs["E"]
        head = f"t = {self.sim.t * m.AU_TIME_FS:7.3f} fs   E = {e:8.4f} Ha ({e * m.HARTREE_EV:7.2f} eV)"
        if two:
            head += f"   electron repulsion V_ee = {obs['Vee']:.4f} Ha"
        if self.sim.nuclei_move:
            head += f"   R = {s.R:.3f} bohr   nuclear KE = {obs['Tnuc']:.5f} Ha"
        levels = "  ".join(f"E_{nm} = {orb[k][1]:.4f}" for k, nm in ((0, "σ"), (1, "σ*")) if k in orb)
        if levels:
            head += f"   one-electron levels: {levels}"
        out["status"] = head + "\n" + "   |   ".join(lines) + "\n" + self.message
        out["molecule_version"] = self.molecule_version
        out["region_version"] = self.region_version
        out["rescale"] = self.molecule_rescale
        self.molecule_rescale = False
        return out
