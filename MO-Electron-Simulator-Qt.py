"""
MO Electron Simulator - pyqtgraph window
----------------------------------------
The same simulator as MO-Electron-Simulator.py (same buttons, sliders and
physics; see that file for what everything means), drawn with pyqtgraph,
which paints straight through Qt instead of re-rendering matplotlib figures.
That leaves more of each frame for the physics and keeps the animation smooth.

Files needed in the same folder: mo_physics.py, mo_controller.py
(and mo_physics_jax.py for --jax).

Setup (once):
    python -m pip install numpy pyqtgraph PyQt6
    (PySide6 or PyQt5 work too; for --jax also: python -m pip install jax)

Run:
    python MO-Electron-Simulator-Qt.py
    python MO-Electron-Simulator-Qt.py --jax        (physics on JAX: CPU or GPU)
    python MO-Electron-Simulator-Qt.py --jax --fp32 (JAX in single precision: faster; best on GPUs)
    python MO-Electron-Simulator-Qt.py --selftest   (physics checks, no window)

Mouse: click in the molecule plot to put an electron there; drag to give it
a velocity.
"""

import sys
import time

import numpy as np

import mo_controller as mc
import mo_physics

APP_TITLE = "MO Electron Simulator"
APP_SUBTITLE = "Electrons between two nuclei: particle ⇄ wavefunction"
HEADER_COLOR = "#1f3b57"


def load_physics(use_jax, fp32=False):
    if use_jax:
        if fp32:
            import os
            os.environ["MO_JAX_FP32"] = "1"          # must be set before JAX is imported
        import mo_physics_jax
        return mo_physics_jax
    return mo_physics


def run_app(physics=mo_physics, show=True):
    """Build the window. Returns (qt_app, window); call qt_app.exec() to run."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    Qt = QtCore.Qt
    backend_name = f"JAX on {physics.device_name()}" if physics.__name__.endswith("jax") else "NumPy"

    class DragViewBox(pg.ViewBox):
        """Molecule plot: no pan/zoom; click places an electron, drag gives it a velocity."""

        def __init__(self, on_move, on_release):
            super().__init__(lockAspect=True, enableMenu=False)
            self.setMouseEnabled(x=False, y=False)
            self.on_move, self.on_release = on_move, on_release

        def mouseDragEvent(self, ev, axis=None):
            if ev.button() != Qt.MouseButton.LeftButton:
                ev.ignore()
                return
            ev.accept()
            start = self.mapSceneToView(ev.buttonDownScenePos())
            cur = self.mapSceneToView(ev.scenePos())
            (self.on_release if ev.isFinish() else self.on_move)(start, cur)

        def mouseClickEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                ev.accept()
                p = self.mapSceneToView(ev.scenePos())
                self.on_release(p, p)

        def wheelEvent(self, ev, axis=None):
            ev.ignore()

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"{APP_TITLE} (pyqtgraph, {backend_name})")
            self.status = None
            self.ctl = mc.Controller(physics, busy=self._busy)
            self.running = True
            self.dragging = False
            self.drawn = dict(molecule=None, region=None)
            self.iso_v, self.iso_region = [], None
            self._build()
            self._sync_widgets()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(mc.FRAME_MS)
            self._fps = [time.perf_counter(), 0, 0.0]

        def _busy(self, text):
            if self.status is not None:
                self.status.setText(text)
                QtWidgets.QApplication.processEvents()

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        def _build(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            outer = QtWidgets.QVBoxLayout(central)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            header = QtWidgets.QLabel(f"  <b style='font-size:15pt'>{APP_TITLE}</b>&nbsp;&nbsp;&nbsp;"
                                      f"{APP_SUBTITLE} &nbsp;&nbsp;[pyqtgraph, physics: {backend_name}]")
            header.setStyleSheet(f"background:{HEADER_COLOR}; color:white; padding:6px;")
            outer.addWidget(header)

            body = QtWidgets.QHBoxLayout()
            outer.addLayout(body, 1)
            self.plots = pg.GraphicsLayoutWidget()
            self.plots.setBackground("w")
            body.addWidget(self.plots, 1)
            self._build_plots()

            panel = QtWidgets.QWidget()
            panel.setFixedWidth(300)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(panel)
            scroll.setWidgetResizable(True)
            scroll.setFixedWidth(318)
            body.addWidget(scroll)
            self._build_panel(panel)

            self.status = QtWidgets.QLabel()
            self.status.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
            self.status.setStyleSheet("background:#eee; border-top:1px solid #aaa; padding:3px 6px;")
            self.status.setTextFormat(Qt.TextFormat.PlainText)
            self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            outer.addWidget(self.status)

        def _build_plots(self):
            ctl = self.ctl
            s = ctl.sys
            lay = self.plots.ci.layout
            vb = DragViewBox(self.on_drag_move, self.on_drag_release)
            self.main = self.plots.addPlot(row=0, col=0, rowspan=4, viewBox=vb)
            self.main.setLabel("bottom", "x (bohr)")
            self.main.setLabel("left", "y (bohr)")
            self.main.hideButtons()
            vb.setBackgroundColor("k")
            hw = mc.VIEW_HALF_WIDTH
            vb.setRange(xRange=(-hw, hw), yRange=(-hw, hw), padding=0)
            self.img = pg.ImageItem()
            x0, x1, y0, y1 = ctl.extent
            self.img_rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)   # grid -> bohr (applied with each image)
            self.main.addItem(self.img)
            self.nuclei = pg.ScatterPlotItem(pen=pg.mkPen("w", width=1.5), brush=pg.mkBrush("#e8483b"))
            self.trails, self.dots = [], []
            for col in mc.ELECTRON_COLORS:
                self.trails.append(self.main.plot([], [], pen=pg.mkPen(col, width=1.4)))
            self.ghost_trail = self.main.plot([], [], pen=pg.mkPen((255, 255, 255, 150), width=1,
                                                                   style=Qt.PenStyle.DashLine))
            self.ghost_dot = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w", width=1.5), brush=None)
            for col in mc.ELECTRON_COLORS:
                d = pg.ScatterPlotItem(size=13, pen=pg.mkPen("w", width=1.2), brush=pg.mkBrush(col))
                self.dots.append(d)
            for item in (self.nuclei, self.ghost_dot, *self.dots):
                self.main.addItem(item)
            self.drag_line = self.main.plot([], [], pen=pg.mkPen("#ffe066", width=2.5))

            small = {"font-size": "8pt"}
            self.cut = self.plots.addPlot(row=0, col=1)
            self.cut.setLabel("left", "energy (Ha)", **small)
            self.cut.addLegend(offset=(-5, 5), labelTextSize="8pt", brush=(255, 255, 255, 190))
            self.cut_v = self.cut.plot(s.x, np.asarray(s.V[s.n // 2]), pen=pg.mkPen("#555", width=1.5),
                                       name="V along bond")
            self.cut_rhos = [self.cut.plot([], [], pen=pg.mkPen(col, width=1.5),
                                           name="|ψ|² along bond" if k == 0 else None)
                             for k, col in enumerate(mc.ELECTRON_LINE_COLORS)]
            self.cut_e = self.cut.plot([], [], pen=pg.mkPen("#333", width=1.2, style=Qt.PenStyle.DotLine),
                                       name="energy")
            self.cut_dots = [pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(col))
                             for col in mc.ELECTRON_LINE_COLORS]
            for d in self.cut_dots:
                self.cut.addItem(d)
            self.cut.setXRange(-hw, hw, padding=0)
            self.cut.enableAutoRange(y=False)

            self.ep = self.plots.addPlot(row=1, col=1)
            self.ep.setLabel("left", "energy (Ha)", **small)
            self.ep.addLegend(offset=(-5, 5), labelTextSize="8pt", brush=(255, 255, 255, 190))
            self.l_t = self.ep.plot([], [], pen=pg.mkPen("#2f9e44", width=1.3), name="kinetic T")
            self.l_v = self.ep.plot([], [], pen=pg.mkPen("#7048e8", width=1.3), name="potential V")
            self.l_e = self.ep.plot([], [], pen=pg.mkPen("k", width=1.8), name="total E")
            self.levels = []
            for col in ("#1c7ed6", "#e8590c"):
                line = pg.InfiniteLine(angle=0, pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DashLine))
                self.ep.addItem(line)
                self.levels.append(line)

            self.inp = self.plots.addPlot(row=2, col=1)
            self.inp.setLabel("left", "inside σ", **small)
            self.inp.addLegend(offset=(-5, 5), labelTextSize="8pt", brush=(255, 255, 255, 190))
            self.inp.setYRange(-0.05, 1.08, padding=0)
            self.l_in = [self.inp.plot([], [], pen=pg.mkPen(col, width=1.4), name=f"electron {k + 1}")
                         for k, col in enumerate(mc.ELECTRON_LINE_COLORS)]

            self.rp = self.plots.addPlot(row=3, col=1)
            self.rp.setLabel("left", "R (bohr)", **small)
            self.rp.setLabel("bottom", "time (fs)", **small)
            self.l_r = self.rp.plot([], [], pen=pg.mkPen("#c92a2a", width=1.5))
            for p in (self.inp, self.rp):
                p.setXLink(self.ep)
            lay.setColumnStretchFactor(0, 5)
            lay.setColumnStretchFactor(1, 4)
            for r, f in enumerate((5, 4, 3, 3)):
                lay.setRowStretchFactor(r, f)

        def _build_panel(self, panel):
            ctl = self.ctl
            v = QtWidgets.QVBoxLayout(panel)
            v.setContentsMargins(4, 4, 4, 4)

            box = QtWidgets.QGroupBox("Electrons")
            g = QtWidgets.QGridLayout(box)
            self.mode_label = QtWidgets.QLabel()
            self.mode_label.setStyleSheet("font-weight:bold;")
            g.addWidget(self.mode_label, 0, 0, 1, 2)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Number:"))
            self.count_group = QtWidgets.QButtonGroup(self)
            for n in (1, 2):
                rb = QtWidgets.QRadioButton(str(n))
                self.count_group.addButton(rb, n)
                row.addWidget(rb)
            self.count_group.idClicked.connect(self.on_count)
            row.addWidget(QtWidgets.QLabel("  Act on:"))
            self.target_group = QtWidgets.QButtonGroup(self)
            self.target_buttons = {}
            for k, (val, text) in enumerate((("1", "e₁"), ("2", "e₂"), ("both", "both"))):
                rb = QtWidgets.QRadioButton(text)
                self.target_group.addButton(rb, k)
                self.target_buttons[val] = rb
                row.addWidget(rb)
            self.target_group.idClicked.connect(lambda k: ctl.set_target(("1", "2", "both")[k]))
            row.addStretch(1)
            g.addLayout(row, 1, 0, 1, 2)
            buttons = [("Become wave ψ", ctl.become_wave), ("Measure (collapse)", ctl.measure),
                       ("Relax → σ bonding", lambda: ctl.start_relax(0)),
                       ("Relax → σ* anti", lambda: ctl.start_relax(1))]
            for k, (text, cmd) in enumerate(buttons):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(lambda _=False, c=cmd: (c(), self.redraw()))
                g.addWidget(b, 2 + k // 2, k % 2)
            self.play_btn = QtWidgets.QPushButton("Pause")
            self.play_btn.clicked.connect(self.toggle_run)
            g.addWidget(self.play_btn, 4, 0)
            b = QtWidgets.QPushButton("Reset")
            b.clicked.connect(self.reset)
            g.addWidget(b, 4, 1)
            v.addWidget(box)

            self.sliders = {}
            for title, sliders in (("Molecule", mc.MOLECULE_SLIDERS), ("Simulation", mc.SIMULATION_SLIDERS)):
                box = QtWidgets.QGroupBox(title)
                g = QtWidgets.QGridLayout(box)
                g.setVerticalSpacing(1)
                for r, (key, label, lo, hi, res) in enumerate(sliders):
                    name = QtWidgets.QLabel(label)
                    val = QtWidgets.QLabel()
                    val.setAlignment(Qt.AlignmentFlag.AlignRight)
                    sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                    sl.setRange(0, int(round((hi - lo) / res)))
                    sl.valueChanged.connect(lambda i, k=key, lo=lo, res=res, lab=val: self.on_slider(k, lo + i * res, lab))
                    g.addWidget(name, 2 * r, 0)
                    g.addWidget(val, 2 * r, 1)
                    g.addWidget(sl, 2 * r + 1, 0, 1, 2)
                    self.sliders[key] = (sl, val, lo, res)
                v.addWidget(box)

            box = QtWidgets.QGroupBox("Options")
            g = QtWidgets.QVBoxLayout(box)
            self.checks = {}
            for key, text in mc.FLAG_LABELS:
                cb = QtWidgets.QCheckBox(text)
                cb.toggled.connect(lambda on, k=key: self.on_flag(k, on))
                g.addWidget(cb)
                self.checks[key] = cb
            v.addWidget(box)
            v.addStretch(1)

        @staticmethod
        def _fmt(value, res):
            return f"{value:.0f}" if res >= 1 else (f"{value:.2f}" if res >= 0.01 else f"{value:.3f}")

        def _sync_widgets(self):
            ctl = self.ctl
            for key, (sl, val, lo, res) in self.sliders.items():
                sl.blockSignals(True)
                sl.setValue(int(round((ctl.params[key] - lo) / res)))
                sl.blockSignals(False)
                val.setText(self._fmt(ctl.params[key], res))
            for key, cb in self.checks.items():
                cb.blockSignals(True)
                cb.setChecked(ctl.flags[key])
                cb.blockSignals(False)
            self.count_group.button(ctl.count).setChecked(True)
            self.target_buttons[ctl.target].setChecked(True)
            for rb in self.target_buttons.values():
                rb.setEnabled(ctl.count == 2)

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------
        def on_slider(self, key, value, label):
            label.setText(self._fmt(value, self.sliders[key][3]))
            self.ctl.set_param(key, value)

        def on_flag(self, key, on):
            self.ctl.set_flag(key, on)
            self._sync_widgets()

        def on_count(self, n):
            self.ctl.set_count(n)
            self._sync_widgets()
            self.redraw()

        def reset(self):
            self.ctl.reset()
            self._sync_widgets()
            self.redraw()

        def toggle_run(self):
            self.running = not self.running
            self.play_btn.setText("Pause" if self.running else "Play")

        def on_drag_move(self, start, cur):
            self.dragging = True
            self.drag_line.setData([start.x(), cur.x()], [start.y(), cur.y()])

        def on_drag_release(self, start, cur):
            self.dragging = False
            self.drag_line.setData([], [])
            self.ctl.place_from_drag(start.x(), start.y(), cur.x(), cur.y())
            self.redraw()

        # ------------------------------------------------------------------
        # Drawing
        # ------------------------------------------------------------------
        def _update_contours(self, f):
            ctl = self.ctl
            s = ctl.sys
            vw = ctl.view
            if f["molecule_version"] != self.drawn["molecule"]:
                self.drawn["molecule"] = f["molecule_version"]
                V = np.asarray(s.V)[vw, vw]
                levels = np.linspace(V.min() * 0.9, -0.15, 9)
                while len(self.iso_v) < len(levels):
                    iso = pg.IsocurveItem(pen=pg.mkPen((255, 255, 255, 80), width=1))
                    iso.setParentItem(self.img)
                    self.iso_v.append(iso)
                for iso, lev in zip(self.iso_v, levels):
                    iso.setData(V, level=lev)
            if f["rescale"]:
                self.cut.setYRange(*f["cut_ylim"], padding=0)
            if f["region_version"] != self.drawn["region"]:
                self.drawn["region"] = f["region_version"]
                dens = ctl.sigma_density()
                if self.iso_region is None:
                    self.iso_region = pg.IsocurveItem(pen=pg.mkPen((255, 255, 255, 220), width=1.6,
                                                                   style=Qt.PenStyle.DashLine))
                    self.iso_region.setParentItem(self.img)
                if dens is not None and ctl.region is not None:
                    self.iso_region.setData(dens[vw, vw], level=ctl.region[1])
                    self.iso_region.setVisible(True)
                else:
                    self.iso_region.setVisible(False)

        def draw(self, f):
            self._update_contours(f)
            self.img.setImage(f["image"], autoLevels=False, levels=(0, 255), rect=self.img_rect)
            self.nuclei.setData([xn for xn, _ in f["nuclei"]], [0, 0],
                                size=[11 + 5 * z for _, z in f["nuclei"]])
            for i in range(2):
                if i < len(f["electrons"]):
                    e = f["electrons"][i]
                    self.dots[i].setData([e["x"]], [e["y"]], symbol="+" if e["wave"] else "o")
                    self.trails[i].setData(*e["trail"])
                else:
                    self.dots[i].setData([], [])
                    self.trails[i].setData([], [])
                line = f["cut_lines"][i]
                self.cut_rhos[i].setData(*(([], []) if line is None else (f["cut_x"], line)))
                dot = f["cut_dots"][i]
                self.cut_dots[i].setData(*(([], []) if dot is None else ([dot[0]], [dot[1]])))
            if f["cut_v"] is not None:
                self.cut_v.setData(f["cut_x"], f["cut_v"])
            e1 = f["cut_energy"]
            self.cut_e.setData(*(([], []) if e1 is None else ([f["cut_x"][0], f["cut_x"][-1]], [e1, e1])))
            gh = f["ghost"]
            self.ghost_trail.setData(*(([], []) if gh is None else gh["trail"]))
            self.ghost_dot.setData(*(([], []) if gh is None else ([gh["x"]], [gh["y"]])))
            if f["title"] != getattr(self, "_title", None):
                self._title = f["title"]
                self.main.setTitle(f["title"], size="10pt")
            self.mode_label.setText(f["mode_label"])
            for line, lev in zip(self.levels, f["levels"]):
                line.setVisible(not np.isnan(lev))
                if not np.isnan(lev):
                    line.setPos(lev)
            h = f["history"]
            if h is not None:
                for line, col in ((self.l_t, 1), (self.l_v, 2), (self.l_e, 3)):
                    line.setData(h[:, 0], h[:, col])
                for k, line in enumerate(self.l_in):
                    ok = ~np.isnan(h[:, 4 + k])
                    line.setData(h[ok, 0], h[ok, 4 + k])
                self.l_r.setData(h[:, 0], h[:, 6])
            else:
                for line in (self.l_t, self.l_v, self.l_e, *self.l_in, self.l_r):
                    line.setData([], [])
            self.status.setText(f["status"])

        def redraw(self):
            self.ctl.last_dt = 0.0               # nothing advanced: no new history point
            self.draw(self.ctl.frame())

        def step_frame(self):
            self.ctl.housekeeping(dragging=self.dragging)
            self.ctl.advance()
            self.draw(self.ctl.frame())

        def tick(self):
            if not self.running:
                return
            try:
                self.step_frame()
            except Exception as exc:               # keep the window alive
                self.running = False
                self.play_btn.setText("Play")
                self.status.setText(f"Error: {exc}")
                raise

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Window()
    win.resize(1320, 840)
    if show:
        win.show()
    return app, win


if __name__ == "__main__":
    physics = load_physics("--jax" in sys.argv, fp32="--fp32" in sys.argv)
    if "--selftest" in sys.argv:
        sys.exit(0 if physics.run_selftest(physics) else 1)
    qt_app, window = run_app(physics)
    sys.exit(qt_app.exec())
