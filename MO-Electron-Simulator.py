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

Vibrating nuclei ("Nuclei vibrate" option):
  The two nuclei become heavy classical particles that move along the bond
  axis (vibration; no rotation). They repel each other (Z1*Z2/R) and are
  pulled by the electrons: by a particle electron's point charge, or by a
  wave electron's charge cloud (Ehrenfest / Hellmann-Feynman forces). Each
  time step the nuclei move, the potential the electrons feel is rebuilt,
  and the electrons respond. The energy shown then includes the nuclei, and
  the bottom-right plot shows the bond length R(t). Real hydrogen nuclei
  weigh 1836 electron masses, so a vibration takes ~10 fs; the mass slider
  makes them lighter to speed it up. Set R away from the balance point
  (R slider) to start a vibration; moving the slider puts the nuclei back at
  rest at the new R.

Mouse: click in the molecule plot to put an electron there; click-and-drag
to also give it a velocity. Waves stay waves, particles stay particles.

Files (keep them in one folder):
    MO-Electron-Simulator.py       this window (tkinter + matplotlib)
    MO-Electron-Simulator-JAX.py   the same window, physics compiled with JAX
    MO-Electron-Simulator-Qt.py    faster window drawn with pyqtgraph
    mo_physics.py                  the physics (NumPy)
    mo_physics_jax.py              the same physics in JAX (CPU or GPU)
    mo_controller.py               buttons/sliders/stepping shared by the windows

Setup (once):
    python -m pip install numpy matplotlib
    tkinter ships with the python.org installers (Linux: sudo apt install python3-tk)

Run:
    python MO-Electron-Simulator.py
    python MO-Electron-Simulator.py --jax          (JAX physics, needs: pip install jax)
    python MO-Electron-Simulator.py --jax --fp32   (JAX in single precision: faster; best on GPUs)
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
  * Nuclei are fixed (Born-Oppenheimer) unless "Nuclei vibrate" is on; then
    they are classical point masses moving along the bond axis. With fixed
    nuclei the reported energies are electronic and the nucleus-nucleus
    repulsion Z1*Z2/R is shown separately; with moving nuclei the energy
    includes their repulsion and kinetic energy.
  * Absorbing edges (optional) remove the part of a wave that leaves the
    box, so its norm is the probability that the electron is still bound.
"""

import sys

import numpy as np

import mo_controller as mc
import mo_physics

# ----------------------------------------------------------------------------
# Window appearance. The simulation settings are in mo_physics.py and
# mo_controller.py.
# ----------------------------------------------------------------------------
APP_TITLE = "MO Electron Simulator"
APP_SUBTITLE = "Electrons between two nuclei: particle ⇄ wavefunction"
HEADER_COLOR = "#1f3b57"
HEADER_TEXT_COLOR = "#ffffff"


def load_physics(use_jax, fp32=False):
    """mo_physics (NumPy) or mo_physics_jax (JAX, CPU or GPU)."""
    if use_jax:
        if fp32:
            import os
            os.environ["MO_JAX_FP32"] = "1"          # must be set before JAX is imported
        import mo_physics_jax
        return mo_physics_jax
    return mo_physics


def run_app(root=None, mainloop=True, physics=mo_physics):
    """Open the simulator window. Pass mainloop=False to get the app object back
    without blocking (used for scripted tests / screenshots)."""
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:                                    # sharper text on Windows high-DPI screens
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    backend_name = f"JAX on {physics.device_name()}" if physics.__name__.endswith("jax") else "NumPy"

    class SimulatorApp:
        def __init__(self, root):
            self.root = root
            root.title(f"{APP_TITLE} ({backend_name})")
            self.status = None
            self.ctl = mc.Controller(physics, busy=self._busy)
            self.running = True
            self.drag_start = None
            self.drawn = dict(molecule=None, region=None)
            self.contours = None
            self.region_art = None
            self._build_ui()
            self._sync_widgets()
            self.root.after(mc.FRAME_MS, self.tick)

        def _busy(self, text):
            if self.status is not None:
                self.status.set(text)
                self.root.update_idletasks()

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        def _build_ui(self):
            root = self.root
            header = tk.Frame(root, bg=HEADER_COLOR)
            header.pack(fill="x")
            tk.Label(header, text=APP_TITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR,
                     font=("TkDefaultFont", 14, "bold")).pack(side="left", padx=12, pady=6)
            tk.Label(header, text=f"{APP_SUBTITLE}   [physics: {backend_name}]", bg=HEADER_COLOR,
                     fg=HEADER_TEXT_COLOR).pack(side="left", padx=6, pady=6)

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
            gs = self.fig.add_gridspec(4, 2, width_ratios=[1.25, 1], height_ratios=[1.25, 1, 0.8, 0.8],
                                       hspace=0.6, wspace=0.25,
                                       left=0.075, right=0.98, top=0.95, bottom=0.07)
            self.ax = self.fig.add_subplot(gs[:, 0])
            self.ax_cut = self.fig.add_subplot(gs[0, 1])
            self.ax_e = self.fig.add_subplot(gs[1, 1])
            self.ax_in = self.fig.add_subplot(gs[2, 1], sharex=self.ax_e)
            self.ax_r = self.fig.add_subplot(gs[3, 1], sharex=self.ax_e)
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
            tk.Label(row, text="  Act on:").pack(side="left")
            self.target_var = tk.StringVar(value="1")
            self.target_buttons = []
            for val, text in (("1", "e₁"), ("2", "e₂"), ("both", "both")):
                b = ttk.Radiobutton(row, text=text, value=val, variable=self.target_var,
                                    command=lambda: self.ctl.set_target(self.target_var.get()))
                b.pack(side="left")
                self.target_buttons.append(b)
            ctl = self.ctl
            buttons = [
                ("Become wave ψ", ctl.become_wave),
                ("Measure (collapse)", ctl.measure),
                ("Relax → σ bonding", lambda: ctl.start_relax(0)),
                ("Relax → σ* anti", lambda: ctl.start_relax(1)),
            ]
            for k, (text, cmd) in enumerate(buttons):
                ttk.Button(box, text=text, command=self._redraw_after(cmd)).grid(
                    row=3 + k // 2, column=k % 2, sticky="ew", padx=2, pady=1)
            self.play_btn = ttk.Button(box, text="Pause", command=self.toggle_run)
            self.play_btn.grid(row=5, column=0, sticky="ew", padx=2, pady=1)
            ttk.Button(box, text="Reset", command=self.reset).grid(
                row=5, column=1, sticky="ew", padx=2, pady=1)

            self.vars = {}
            for title, sliders in (("Molecule", mc.MOLECULE_SLIDERS), ("Simulation", mc.SIMULATION_SLIDERS)):
                box = ttk.LabelFrame(panel, text=title)
                box.pack(fill="x", pady=3)
                for key, label, lo, hi, res in sliders:
                    self._slider(box, key, label, lo, hi, res)

            box = ttk.LabelFrame(panel, text="Options")
            box.pack(fill="x", pady=3)
            self.flag_vars = {}
            for key, text in mc.FLAG_LABELS:
                v = tk.BooleanVar(value=mc.FLAG_DEFAULTS[key])
                ttk.Checkbutton(box, text=text, variable=v,
                                command=lambda k=key, v=v: self.on_flag(k, v.get())).pack(anchor="w")
                self.flag_vars[key] = v

        def _slider(self, parent, key, label, lo, hi, res):
            var = tk.DoubleVar(value=self.ctl.params[key])
            self.vars[key] = var
            row = tk.Frame(parent)
            row.pack(fill="x", padx=4)
            tk.Label(row, text=label).pack(side="left")
            tk.Label(row, textvariable=var, width=6, anchor="e").pack(side="right")
            tk.Scale(parent, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var,
                     showvalue=0, width=10, sliderlength=18, length=220,
                     command=lambda _v, k=key, v=var: self.ctl.set_param(k, v.get())
                     ).pack(fill="x", padx=4, pady=(0, 2))

        def _sync_widgets(self):
            """Copy the controller's settings into the widgets (after Reset etc.)."""
            for k, v in self.vars.items():
                if abs(v.get() - self.ctl.params[k]) > 1e-12:
                    v.set(self.ctl.params[k])
            for k, v in self.flag_vars.items():
                v.set(self.ctl.flags[k])
            self.count_var.set(self.ctl.count)
            for b in self.target_buttons:
                b.state(["!disabled"] if self.ctl.count == 2 else ["disabled"])

        def _build_axes(self):
            s = self.ctl.sys
            ax = self.ax
            n_view = self.ctl.view_x.size
            self.img = ax.imshow(np.zeros((n_view, n_view, 3), np.uint8), origin="lower",
                                 extent=self.ctl.extent, interpolation="bilinear")
            ax.set_xlim(-mc.VIEW_HALF_WIDTH, mc.VIEW_HALF_WIDTH)
            ax.set_ylim(-mc.VIEW_HALF_WIDTH, mc.VIEW_HALF_WIDTH)
            ax.set_xlabel("x (bohr)")
            ax.set_ylabel("y (bohr)")
            self.title = ax.set_title("", fontsize=10)
            self.nuclei_art = ax.scatter([], [], s=[], c="#e8483b", edgecolors="white",
                                         linewidths=1.2, zorder=5)
            self.trails, self.dots = [], []
            for col in mc.ELECTRON_COLORS:
                (tr,) = ax.plot([], [], "-", color=col, lw=1.2, alpha=0.85, zorder=6)
                (dot,) = ax.plot([], [], "o", color=col, ms=9, mec="white", zorder=7)
                self.trails.append(tr)
                self.dots.append(dot)
            (self.ghost_trail,) = ax.plot([], [], "--", color="white", lw=0.9, alpha=0.6, zorder=6)
            (self.ghost_dot,) = ax.plot([], [], "o", mfc="none", mec="white", ms=9, zorder=7)
            (self.drag_line,) = ax.plot([], [], "-", color="#ffe066", lw=2, zorder=8)

            a = self.ax_cut
            a.set_ylabel("energy (Ha)")
            a.tick_params(labelsize=8)
            (self.cut_v,) = a.plot(s.x, np.asarray(s.V[s.n // 2]), color="#555", lw=1.5, label="V along bond")
            self.cut_rhos, self.cut_dots = [], []
            for k, col in enumerate(mc.ELECTRON_LINE_COLORS):
                (line,) = a.plot([], [], color=col, lw=1.5, label="|ψ|² along bond" if k == 0 else None)
                (dot,) = a.plot([], [], "o", color=col, ms=7)
                self.cut_rhos.append(line)
                self.cut_dots.append(dot)
            (self.cut_e,) = a.plot([], [], ":", color="#333", lw=1.2, label="energy")
            a.set_xlim(-mc.VIEW_HALF_WIDTH, mc.VIEW_HALF_WIDTH)
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_e
            a.set_ylabel("energy (Ha)")
            a.tick_params(labelbottom=False)
            (self.l_t,) = a.plot([], [], color="#2f9e44", lw=1.2, label="kinetic T")
            (self.l_v,) = a.plot([], [], color="#7048e8", lw=1.2, label="potential V")
            (self.l_e,) = a.plot([], [], color="black", lw=1.6, label="total E")
            self.level_lines = [a.axhline(np.nan, color=c, ls="--", lw=0.9) for c in ("#1c7ed6", "#e8590c")]
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_in
            a.tick_params(labelbottom=False)
            a.set_ylabel("inside σ")
            a.set_ylim(-0.05, 1.08)
            self.l_in = [a.plot([], [], color=col, lw=1.3, label=f"electron {k + 1}")[0]
                         for k, col in enumerate(mc.ELECTRON_LINE_COLORS)]
            a.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                     frameon=False, handlelength=1.5, columnspacing=1.0)

            a = self.ax_r
            a.set_xlabel("time (fs)")
            a.set_ylabel("R (bohr)")
            (self.l_r,) = a.plot([], [], color="#c92a2a", lw=1.4)

            # Blitting: everything that changes every frame is "animated" and redrawn
            # on top of a cached picture of the static parts (axes, ticks, labels).
            self.anim = [self.nuclei_art, *self.trails, self.ghost_trail, *self.dots,
                         self.ghost_dot, self.drag_line, self.title,
                         *self.cut_rhos, *self.cut_dots, self.cut_e,
                         self.l_t, self.l_v, self.l_e, *self.level_lines, *self.l_in, self.l_r, self.cut_v]
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

        # ------------------------------------------------------------------
        # Drawing a frame
        # ------------------------------------------------------------------
        def _update_contours(self, f):
            s = self.ctl.sys
            if f["molecule_version"] != self.drawn["molecule"]:
                self.drawn["molecule"] = f["molecule_version"]
                if self.contours is not None:
                    self.contours.remove()
                V = np.asarray(s.V)
                levels = np.linspace(V.min() * 0.9, -0.15, 9)
                self.contours = self.ax.contour(s.X, s.Y, V, levels=levels, colors="white",
                                                linewidths=0.5, alpha=0.3)
                self.contours.set_animated(True)
                self.nuclei_art.set_sizes([60 + 60 * z for _, z in s.nuclei])
            if f["rescale"]:
                self.ax_cut.set_ylim(*f["cut_ylim"])
                self.need_full_draw = True
            if f["region_version"] != self.drawn["region"]:
                self.drawn["region"] = f["region_version"]
                if self.region_art is not None:
                    self.region_art.remove()
                    self.region_art = None
                dens = self.ctl.sigma_density()
                if dens is not None and self.ctl.region is not None:
                    self.region_art = self.ax.contour(s.X, s.Y, dens, levels=[self.ctl.region[1]],
                                                      colors="white", linewidths=1.0,
                                                      linestyles="--", alpha=0.8)
                    self.region_art.set_animated(True)

        def _update_limits(self, h):
            """Rescale the time plots only when the data leaves them (rescaling = full redraw)."""
            ax = self.ax_e
            t0, t1 = h[0, 0], h[-1, 0]
            lo, hi = float(np.nanmin(h[:, 1:4])), float(np.nanmax(h[:, 1:4]))
            for lev in self.frame_levels:
                if not np.isnan(lev):
                    lo, hi = min(lo, lev), max(hi, lev)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            span = max(hi - lo, 0.2)
            if t1 > x1 or t0 < x0:
                ax.set_xlim(t0, t0 + max(1.5 * (t1 - t0), 0.01))
                self.need_full_draw = True
            if lo < y0 or hi > y1 or span < 0.3 * (y1 - y0):
                ax.set_ylim(lo - 0.15 * span, hi + 0.35 * span)
                self.need_full_draw = True
            r = h[:, 6]
            rlo, rhi = float(r.min()), float(r.max())
            pad = max(0.15, 0.2 * (rhi - rlo))
            r0, r1 = self.ax_r.get_ylim()
            if rlo < r0 or rhi > r1 or (rhi - rlo + 2 * pad) < 0.4 * (r1 - r0):
                self.ax_r.set_ylim(rlo - pad, rhi + pad)
                self.need_full_draw = True

        def draw(self, f):
            self._update_contours(f)
            self.img.set_data(f["image"])
            self.nuclei_art.set_offsets([(xn, 0) for xn, _ in f["nuclei"]])
            for i in range(2):
                if i < len(f["electrons"]):
                    e = f["electrons"][i]
                    self.dots[i].set_data([e["x"]], [e["y"]])
                    self.dots[i].set_marker("P" if e["wave"] else "o")
                    self.trails[i].set_data(*e["trail"])
                else:
                    self.dots[i].set_data([], [])
                    self.trails[i].set_data([], [])
                line = f["cut_lines"][i]
                self.cut_rhos[i].set_data(([], []) if line is None else (f["cut_x"], line))
                dot = f["cut_dots"][i]
                self.cut_dots[i].set_data(([], []) if dot is None else ([dot[0]], [dot[1]]))
            if f["cut_v"] is not None:
                self.cut_v.set_ydata(f["cut_v"])
            e1 = f["cut_energy"]
            self.cut_e.set_data(([], []) if e1 is None else ([f["cut_x"][0], f["cut_x"][-1]], [e1, e1]))
            g = f["ghost"]
            self.ghost_trail.set_data(*(([], []) if g is None else g["trail"]))
            self.ghost_dot.set_data(([], []) if g is None else ([g["x"]], [g["y"]]))
            self.title.set_text(f["title"])
            self.mode_var.set(f["mode_label"])

            self.frame_levels = f["levels"]
            for line, lev in zip(self.level_lines, f["levels"]):
                line.set_ydata([lev, lev])
            h = f["history"]
            if h is not None:
                for line, col in ((self.l_t, 1), (self.l_v, 2), (self.l_e, 3)):
                    line.set_data(h[:, 0], h[:, col])
                for k, line in enumerate(self.l_in):
                    line.set_data(h[:, 0], h[:, 4 + k])
                self.l_r.set_data(h[:, 0], h[:, 6])
                self._update_limits(h)
            else:
                for line in (self.l_t, self.l_v, self.l_e, *self.l_in, self.l_r):
                    line.set_data([], [])
            self.status.set(f["status"])
            self._refresh_canvas()

        def redraw(self):
            self.ctl.last_dt = 0.0               # nothing advanced: no new history point
            self.draw(self.ctl.frame())

        def _redraw_after(self, action):
            def run():
                action()
                self.redraw()
            return run

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------
        def reset(self):
            self.ctl.reset()
            self._sync_widgets()
            self.redraw()

        def on_count(self):
            self.ctl.set_count(self.count_var.get())
            self._sync_widgets()
            self.redraw()

        def on_flag(self, key, value):
            self.ctl.set_flag(key, value)
            self.need_full_draw = True
            self._sync_widgets()

        def toggle_run(self):
            self.running = not self.running
            self.play_btn.config(text="Pause" if self.running else "Play")

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
            self.ctl.place_from_drag(x0, y0, x1, y1)
            self.redraw()

        def step_frame(self):
            """One animation frame: deferred work, simulation, drawing."""
            self.ctl.housekeeping(dragging=self.drag_start is not None)
            self.ctl.advance()
            self.draw(self.ctl.frame())

        def tick(self):
            if self.running:
                try:
                    self.step_frame()
                except Exception as exc:           # keep the window alive
                    self.running = False
                    self.ctl.message = f"Error: {exc}"
                    self.play_btn.config(text="Play")
                    self.status.set(f"Error: {exc}")
            self.root.after(mc.FRAME_MS, self.tick)

    root = root or tk.Tk()
    app = SimulatorApp(root)
    if mainloop:
        root.mainloop()
    return app


if __name__ == "__main__":
    physics = load_physics("--jax" in sys.argv, fp32="--fp32" in sys.argv)
    if "--selftest" in sys.argv:
        sys.exit(0 if physics.run_selftest(physics) else 1)
    run_app(physics=physics)
