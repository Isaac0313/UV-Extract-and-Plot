"""
UV-Vis Plotter
--------------
Open an Excel (or CSV) file and graph it: one column is X (wavelength),
every other column is a Y series. Y columns can be moved into a
"Working list", where functions (normalize, baseline, smooth, ...) are
applied to them.

Setup (once):
    python -m pip install pandas matplotlib openpyxl scipy

Run:
    python uvvis_plotter.py
    python uvvis_plotter.py mydata.xlsx      (optional: open a file right away)

Expected layout: the first row holds the column names, the data is below it.

Where the data lives (all in memory, nothing is written unless you export):
    self.df        the sheet you are viewing (a pandas table, like the spreadsheet)
    self.ws.items  the working list: {name: {"x": array, "y": array, "ops": [...]}}
                   these are COPIES; functions change them, never your Excel file.
    self.log       every function applied (export it with "Export function log")
"""

import copy
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ----------------------------------------------------------------------------
# Header shown at the top of the window. Edit these lines to change it;
# it can't be changed from inside the program.
# ----------------------------------------------------------------------------
APP_TITLE = "UV-Vis Plotter"
APP_SUBTITLE = "Spectra plotting and parameter extraction"
HEADER_COLOR = "#1f3b57"        # background of the header band
HEADER_TEXT_COLOR = "#ffffff"

# Columns of the function log CSV
LOG_COLUMNS = ["Time", "File", "Action", "Item", "Value", "Result",
               "Peak", "Center", "Height", "FWHM", "Area", "Baseline", "Eg", "R2"]

STYLES = {
    "Line": dict(linestyle="-", marker=""),
    "Points": dict(linestyle="", marker="o", markersize=3),
    "Line + points": dict(linestyle="-", marker="o", markersize=3),
}


# ============================================================================
# Functions for the working list
# ----------------------------------------------------------------------------
# An "each" function takes (x, y, value) and returns a new (x, y).
#   x, y   numpy arrays, sorted by x (wavelength), blanks already removed
#   value  the text typed in the Value box ("" if the box is unused)
# A "multi" function takes (ws, keys, value), can read/create several items,
# and may return text for the Results box.
# To add your own: write a function like the ones below, then add one entry
# to FUNCTIONS at the bottom of this section. It shows up in the dropdown.
# ============================================================================
def _numbers(value):
    """'400, 500' -> [400.0, 500.0]; '' -> []."""
    parts = [p for p in str(value or "").replace(";", ",").split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Couldn't read '{value}' as numbers. Example: 400, 500")


def _num(value, what="a value"):
    nums = _numbers(value)
    if len(nums) != 1:
        raise ValueError(f"Enter a single number for {what}.")
    return nums[0]


# ---- Add offset, Smooth, 1st derivative (applied to each item) ------------
def fn_offset(x, y, value):
    """Shift the curve up (or down, with a negative number)."""
    return x, y + _num(value, "the offset")


def fn_smooth(x, y, value):
    """Centered moving average over N points."""
    n = int(round(_num(value, "the window size")))
    if n < 2:
        raise ValueError("The window must be at least 2 points.")
    return x, pd.Series(y).rolling(n, center=True, min_periods=1).mean().to_numpy()


def fn_derivative(x, y, value):
    """dY/dX at each point."""
    if len(x) < 2:
        raise ValueError("Need at least 2 points for a derivative.")
    return x, np.gradient(y, x)


# ---- Average (several items -> one new item) -------------------------------
def multi_average(ws, keys, value):
    """Average the items into a new item, on the X points of the first one,
    over the X range they all share."""
    if len(keys) < 2:
        raise ValueError("Needs at least 2 items to average.")
    items = [ws.items[k] for k in keys]
    if len({it.get("tauc", "") for it in items}) > 1:
        raise ValueError("Can't average Tauc plot items with spectra (different X units).")
    lo = max(it["x"].min() for it in items)
    hi = min(it["x"].max() for it in items)
    x = items[0]["x"][(items[0]["x"] >= lo) & (items[0]["x"] <= hi)]
    if len(x) == 0:
        raise ValueError("These items don't share any X range.")
    y = np.mean([np.interp(x, it["x"], it["y"]) for it in items], axis=0)
    new = ws.new_item(f"Average of {len(keys)}", x, y, tauc=items[0].get("tauc", ""))
    ws.details.append(dict(item=", ".join(keys), result=f"created '{new}'"))


# ---- Deconvolution (Gaussian peak fitting) ---------------------------------
# Models the curve as a flat baseline plus N Gaussian peaks:
#     y = c + sum_i  A_i * exp(-(x - center_i)^2 / (2 * sigma_i^2))
# and finds the A, center, sigma and c that fit the data best (least squares).
GAUSS_FWHM = 2 * np.sqrt(2 * np.log(2))     # FWHM = 2.355 * sigma


def _gauss(x, amp, center, sigma):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def _gauss_sum(x, *p):
    """p = (c, A1, center1, sigma1, A2, center2, sigma2, ...)"""
    y = np.full_like(x, p[0], dtype=float)
    for i in range(1, len(p), 3):
        y += _gauss(x, *p[i:i + 3])
    return y


def _auto_peak_guesses(x, y, max_peaks=6):
    """Find likely peaks, including shoulders, from dips in the 2nd derivative
    of a smoothed curve. Returned strongest first."""
    from scipy.signal import find_peaks, savgol_filter
    win = max(7, len(y) // 15) | 1                       # odd window, ~7% of the points
    if win >= len(y):
        return [float(x[int(np.argmax(y))])]
    smooth = savgol_filter(y, win, 3)
    curv = -savgol_filter(y, win, 3, deriv=2)            # big where peaks are
    idx, _ = find_peaks(curv, prominence=0.1 * np.abs(curv).max())
    floor = smooth.min() + 0.05 * (smooth.max() - smooth.min())
    idx = [i for i in idx if smooth[i] > floor]          # ignore noise at the floor
    if not idx:
        return [float(x[int(np.argmax(smooth))])]
    idx = sorted(idx, key=lambda i: -curv[i])[:max_peaks]
    return [float(x[i]) for i in idx]


def multi_deconvolve(ws, keys, value):
    """Fit each item with Gaussian peaks. Adds the peaks and the total fit to
    the working list (dashed) and writes a table to the Results box."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        raise ValueError("Deconvolution needs SciPy. Install it with:\n"
                         "python -m pip install scipy")

    def fit(key, x, y, centers):
        xr = x.max() - x.min()
        base = float(np.min(y))
        dx = float(np.median(np.diff(x)))
        min_sigma = max(dx, 0.01 * xr)      # UV-Vis bands are broad; stops fitting noise spikes
        sigma0 = max(xr / (6 * len(centers)), 2 * min_sigma)
        # starting guesses and limits for: baseline, then (height, center, sigma) per peak
        p0, lo, hi = [base], [-np.inf], [np.inf]
        for c in centers:
            p0 += [max(np.interp(c, x, y) - base, 1e-9), c, sigma0]
            lo += [0, x.min(), min_sigma]
            hi += [np.inf, x.max(), xr]
        try:
            return curve_fit(_gauss_sum, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)[0]
        except RuntimeError:
            raise ValueError(f"The fit for '{key}' didn't converge. Try typing peak "
                             f"positions in the Value box, or fewer peaks.")

    report = []
    for key in keys:
        it = ws.items[key]
        x, y = it["x"], it["y"]
        typed = _numbers(value)
        centers = typed or _auto_peak_guesses(x, y)[:max(1, (len(x) - 2) // 3)]
        bad = [c for c in centers if not x.min() <= c <= x.max()]
        if bad:
            raise ValueError(f"Peak guess {bad[0]:g} is outside the data of '{key}' "
                             f"({x.min():g} to {x.max():g}).")
        if len(x) < 3 * len(centers) + 2:
            raise ValueError(f"'{key}' has too few points for {len(centers)} peaks.")

        if typed:
            p = fit(key, x, y, centers)
        else:
            # auto mode: try the 1, 2, 3... strongest candidates and keep the
            # simplest model that fits well (lowest BIC: rewards a good fit,
            # penalizes every extra peak), so noise doesn't become fake peaks
            best, best_bic = None, np.inf
            for n in range(1, len(centers) + 1):
                try:
                    trial = fit(key, x, y, centers[:n])
                except ValueError:
                    continue
                heights = trial[1::3]
                tiny = heights < 0.02 * heights.max()
                if tiny.any():                      # drop negligible peaks and refit
                    try:
                        trial = fit(key, x, y, list(trial[2::3][~tiny]))
                    except ValueError:
                        continue
                ss = max(np.sum((y - _gauss_sum(x, *trial)) ** 2), 1e-300)
                bic = len(x) * np.log(ss / len(x)) + len(trial) * np.log(len(x))
                if bic < best_bic:
                    best, best_bic = trial, bic
            if best is None:
                raise ValueError(f"The fit for '{key}' didn't converge. Try typing "
                                 f"peak positions in the Value box.")
            p = best

        total = _gauss_sum(x, *p)
        ss_res = np.sum((y - total) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        peaks = sorted((p[i:i + 3] for i in range(1, len(p), 3)), key=lambda q: q[1])
        lines = [f"{key}: {len(peaks)} peak(s), baseline = {p[0]:.4g}, R^2 = {r2:.5f}",
                 f"  {'#':>2} {'center':>9} {'height':>10} {'FWHM':>9} {'area':>10}"]
        for n, (amp, center, sigma) in enumerate(peaks, 1):
            area = amp * abs(sigma) * np.sqrt(2 * np.pi)
            ws.details.append(dict(item=key, result=f"peak {n} of {len(peaks)}",
                                   peak=n, center=center, height=amp,
                                   fwhm=GAUSS_FWHM * abs(sigma), area=area,
                                   baseline=p[0], r2=r2))
            lines.append(f"  {n:>2} {center:>9.2f} {amp:>10.4g} "
                         f"{GAUSS_FWHM * abs(sigma):>9.2f} {area:>10.4g}")
            ws.new_item(f"{key} peak {n} @ {center:.0f}",          # drawn on the baseline
                        x, p[0] + _gauss(x, amp, center, sigma), dashed=True,
                        tauc=it.get("tauc", ""))
        ws.new_item(f"{key} fit", x, total, dashed=True, tauc=it.get("tauc", ""))
        report.append("\n".join(lines))
    return "\n".join(report)


# ---- Tauc plot (optical band gap) ------------------------------------------
# Turns a spectrum (X = wavelength in nm, Y = absorbance) into a Tauc plot:
#     X = photon energy  E = 1239.84 / wavelength   (eV)
#     Y = (A * E)^2      direct allowed transition
#     Y = (A * E)^(1/2)  indirect allowed transition
# Absorbance stands in for the absorption coefficient; they differ only by a
# constant factor, which doesn't move the band gap.
# A straight line fitted over the energy range you type is extended down to
# Y = 0; where it crosses the axis is the band gap Eg.
HC_EV_NM = 1239.84193
TAUC_POWER = {"direct": 2.0, "indirect": 0.5}


def multi_tauc(ws, keys, value, kind):
    """Make a Tauc plot of each item (drawn in a second graph below the
    spectra). With an energy range typed, also fit the line and report Eg.
    Applied to an existing Tauc plot item, it only does the fit."""
    rng = _numbers(value)
    if rng and len(rng) != 2:
        raise ValueError("Type two energies for the fit range, e.g. 2.9, 3.2 (in eV).")
    report = []
    for key in keys:
        it = ws.items[key]
        if it.get("tauc"):
            if it["tauc"] != kind:
                raise ValueError(f"'{key}' is already a Tauc plot ({it['tauc']}). "
                                 f"Use the original spectrum to make a {kind} one.")
            if not rng:
                raise ValueError(f"'{key}' is already a Tauc plot. Type the energy "
                                 f"range to fit, e.g. 2.9, 3.2.")
            name, e, t = key, it["x"], it["y"]
        else:
            if it["x"].min() <= 0:
                raise ValueError(f"'{key}' has X values of 0 or below. The X column "
                                 f"must be wavelength in nm.")
            e = HC_EV_NM / it["x"]
            t = np.clip(it["y"] * e, 0, None) ** TAUC_POWER[kind]
            order = np.argsort(e)
            e, t = e[order], t[order]
            name = ws.new_item(f"{key} Tauc {kind}", e, t, tauc=kind)
            ws.details.append(dict(item=key, result=f"created '{name}'"))
        if not rng:
            continue

        lo, hi = sorted(rng)
        sel = (e >= lo) & (e <= hi)
        if sel.sum() < 3:
            raise ValueError(f"Only {sel.sum()} point(s) of '{key}' are between {lo:g} "
                             f"and {hi:g} eV. Widen the range (the data covers "
                             f"{e.min():.3g} to {e.max():.3g} eV).")
        slope, icpt = np.polyfit(e[sel], t[sel], 1)
        if slope <= 0:
            raise ValueError(f"The line fitted to '{key}' between {lo:g} and {hi:g} eV "
                             f"doesn't rise, so it gives no band gap. Pick the straight, "
                             f"rising part of the absorption edge.")
        eg = -icpt / slope
        ss_res = np.sum((t[sel] - (slope * e[sel] + icpt)) ** 2)
        ss_tot = np.sum((t[sel] - t[sel].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        line_x = np.linspace(eg, hi, 50)                  # from Y = 0 up to the range
        ws.new_item(f"{name} fit", line_x, slope * line_x + icpt, dashed=True, tauc=kind)
        ws.details.append(dict(item=name, result=f"Eg = {eg:.3f} eV", eg=eg, r2=r2))
        report.append(f"{name}: Eg = {eg:.3f} eV  (fit {lo:g}-{hi:g} eV, "
                      f"{sel.sum()} points, R^2 = {r2:.4f})")
    return "\n".join(report) or None


# ---- the dropdown ------------------------------------------------------------
# kind:  "each" = applied to every item, "multi" = uses several items / makes new ones
# hint:  label for the Value box (None = box not used)
# tag:   added to the item's name so you can see what was done; {v} = value
FUNCTIONS = {
    "Average":                  dict(kind="multi", fn=multi_average, hint=None, default="", tag=""),
    "Deconvolution (Gaussian)": dict(kind="multi", fn=multi_deconvolve,
                                     hint="Peak positions, e.g. 480, 550 (blank = auto)",
                                     default="", tag=""),
    "Tauc plot (direct)":       dict(kind="multi", fn=lambda ws, k, v: multi_tauc(ws, k, v, "direct"),
                                     hint="Fit range in eV, e.g. 2.9, 3.2 (blank = plot only)",
                                     default="", tag=""),
    "Tauc plot (indirect)":     dict(kind="multi", fn=lambda ws, k, v: multi_tauc(ws, k, v, "indirect"),
                                     hint="Fit range in eV, e.g. 2.9, 3.2 (blank = plot only)",
                                     default="", tag=""),
    "Add offset":               dict(kind="each", fn=fn_offset, hint="Offset", default="0.1", tag="+{v}"),
    "Smooth (moving average)":  dict(kind="each", fn=fn_smooth, hint="Window (number of points)", default="5", tag="smooth {v}"),
    "1st derivative":           dict(kind="each", fn=fn_derivative, hint=None, default="", tag="d/dx"),
}


# ============================================================================
# The working list (no GUI code here)
# ============================================================================
class WorkingSet:
    """Copies of Y columns that functions are applied to."""

    def __init__(self):
        self.items = {}       # name -> {"x", "y", "ops": [tags], "dashed": bool, "tauc": str}
        self._history = []    # snapshots for Undo: (items, extra)
        self.details = []     # filled by functions with numbers for the log

    def keys(self):
        return list(self.items)

    def label(self, key):
        ops = self.items[key]["ops"]
        return f"{key}  [{', '.join(ops)}]" if ops else key

    def sources(self):
        """Names of the spreadsheet columns currently in the working list."""
        return {it.get("source") for it in self.items.values()}

    def rename(self, old, new):
        """Give an item a new name (keeps its place in the list). The new name
        replaces the [function] tags, so the legend shows exactly what you typed."""
        new = str(new).strip()
        if not new:
            raise ValueError("The name can't be empty.")
        if new != old and new in self.items:
            raise ValueError(f"There is already an item called '{new}'.")
        self.items = {(new if k == old else k): v for k, v in self.items.items()}
        self.items[new]["ops"] = []
        self.items[new]["renamed"] = True

    def unique_key(self, base):
        key, n = base, 2
        while key in self.items:
            key, n = f"{base} #{n}", n + 1
        return key

    def checkpoint(self, extra=None):
        """Save the current state for Undo. 'extra' is anything else the
        caller wants back on undo (the app stores the Y column lists)."""
        self._history.append((copy.deepcopy(self.items), extra))
        del self._history[:-50]               # keep the last 50 steps

    def can_undo(self):
        return bool(self._history)

    def undo(self):
        """Go back one step. Returns the 'extra' saved with that step."""
        if self._history:
            self.items, extra = self._history.pop()
            return extra
        return None

    def add(self, key, x, y):
        """Copy a column in: numbers only, sorted by X, duplicate X removed.
        Returns the item's name (made unique if 'key' is already taken)."""
        x = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(float)
        y = pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy(float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok], y[ok]
        if len(x) == 0:
            raise ValueError(f"'{key}' has no numeric data to move.")
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        x, first = np.unique(x, return_index=True)
        name = self.unique_key(key)
        self.items[name] = dict(x=x, y=y[first], ops=[], dashed=False, source=key)
        return name

    def new_item(self, base_name, x, y, dashed=False, tauc=""):
        """Add a result (average, fitted peak...) as a new item.
        tauc = "direct"/"indirect" for Tauc plot items (X is photon energy)."""
        key = self.unique_key(base_name)
        self.items[key] = dict(x=np.asarray(x, float), y=np.asarray(y, float),
                               ops=[], dashed=dashed, tauc=tauc)
        return key

    def remove(self, keys):
        for k in keys:
            self.items.pop(k, None)

    def apply(self, name, keys, value=""):
        """Run a function from FUNCTIONS. Returns text for Results, or None."""
        if not keys:
            raise ValueError("The working list is empty. Move some Y columns into it first.")
        spec = FUNCTIONS[name]
        value = str(value or "").strip()
        report = None
        self.details = []

        self.checkpoint()
        try:
            if spec["kind"] == "multi":
                report = spec["fn"](self, keys, value)
            else:
                # compute everything first so one failure leaves nothing half-done
                results = {k: spec["fn"](self.items[k]["x"], self.items[k]["y"], value)
                           for k in keys}
                tag = spec["tag"].format(v=value)
                for k, (x, y) in results.items():
                    self.items[k]["x"] = np.asarray(x, float)
                    self.items[k]["y"] = np.asarray(y, float)
                    self.items[k]["ops"].append(tag)
        except Exception:
            self.items = self._history.pop()[0]   # roll back
            raise
        return report

    def to_frame(self):
        """Table for export: an X and a Y column per item (X ranges can differ)."""
        cols = {}
        for k, it in self.items.items():
            cols[f"{self.label(k)} X"] = pd.Series(it["x"])
            cols[f"{self.label(k)} Y"] = pd.Series(it["y"])
        return pd.DataFrame(cols)


# ============================================================================
# Loading and drawing (no GUI code here)
# ============================================================================
def load_tables(path):
    """Read a file and return {sheet_name: DataFrame}."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".tsv"):
        # sep=None lets pandas work out commas vs tabs vs semicolons
        tables = {"Data": pd.read_csv(path, sep=None, engine="python")}
    else:
        tables = pd.read_excel(path, sheet_name=None)

    cleaned = {}
    for name, df in tables.items():
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        names = []
        for c in df.columns:
            base = name_ = str(c).strip()
            n = 2
            while name_ in names:
                name_, n = f"{base} #{n}", n + 1
            names.append(name_)
        df.columns = names
        if df.shape[1] >= 2 and len(df) > 0:
            cleaned[str(name)] = df
    return cleaned


def draw_plot(ax, series, style="Line", log_y=False, grid=True,
              title="", x_label="", y_label="", xlim=(None, None), ylim=(None, None)):
    """series = list of (label, x, y, look), look = "original", "working" or "fit".
    xlim / ylim = (min, max); None = automatic. min > max flips the axis.
    Returns number of lines drawn."""
    ax.clear()
    drawn = 0
    for label, x, y, look in series:
        x = pd.to_numeric(pd.Series(list(x)), errors="coerce")
        y = pd.to_numeric(pd.Series(list(y)), errors="coerce")
        ok = x.notna() & y.notna()          # skip blanks and text cells
        if ok.sum() == 0:
            continue
        if look == "fit":                    # deconvolution peaks / total fit
            ax.plot(x[ok], y[ok], label=label, linewidth=1.2, linestyle="--")
        else:
            ax.plot(x[ok], y[ok], label=label,
                    linewidth=2.0 if look == "working" else 1.2, **STYLES[style])
        drawn += 1

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label or (series[0][0] if len(series) == 1 else "Value"))
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
        if ylim[0] is not None and ylim[0] <= 0:
            ylim = (None, ylim[1])            # log axis can't start at 0 or below
    if xlim != (None, None):
        ax.set_xlim(*xlim)
    if ylim != (None, None):
        ax.set_ylim(*ylim)
    ax.grid(grid, alpha=0.3)
    if drawn:
        ax.legend(fontsize=8 if drawn > 8 else 9, ncol=2 if drawn > 12 else 1)
    return drawn


# ============================================================================
# Interface
# ============================================================================
class App(tk.Tk):
    def __init__(self, start_file=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1450x880")
        self.minsize(1100, 650)

        self.tables = {}
        self.df = None
        self.y_names = []          # columns shown in the "available" list
        self.ws = WorkingSet()
        self.log = []              # rows for the function log CSV
        self.file_name = ""

        self._build_header()
        self._build_left_panel()
        self._build_right_panel()
        self._build_plot_area()

        if start_file:
            self.open_file(start_file)

    # ---- layout -----------------------------------------------------------
    def _build_header(self):
        """Fixed title band. Text comes from APP_TITLE / APP_SUBTITLE at the top."""
        band = tk.Frame(self, bg=HEADER_COLOR, padx=14, pady=8)
        band.pack(side="top", fill="x")
        tk.Label(band, text=APP_TITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR,
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        if APP_SUBTITLE:
            tk.Label(band, text=APP_SUBTITLE, bg=HEADER_COLOR, fg=HEADER_TEXT_COLOR,
                     font=("Segoe UI", 10)).pack(side="left", padx=(14, 0), pady=(5, 0))

    def _build_left_panel(self):
        side = ttk.Frame(self, padding=10)
        side.pack(side="left", fill="y")

        ttk.Button(side, text="Open Excel / CSV file...",
                   command=self.choose_file).pack(fill="x")
        self.file_label = ttk.Label(side, text="No file loaded", wraplength=230,
                                    foreground="#666")
        self.file_label.pack(fill="x", pady=(4, 8))

        ttk.Label(side, text="Sheet").pack(anchor="w")
        self.sheet_box = ttk.Combobox(side, state="readonly", width=30)
        self.sheet_box.pack(fill="x", pady=(0, 6))
        self.sheet_box.bind("<<ComboboxSelected>>", lambda e: self.on_sheet_change())

        ttk.Label(side, text="X column (wavelength)").pack(anchor="w")
        self.x_box = ttk.Combobox(side, state="readonly", width=30)
        self.x_box.pack(fill="x", pady=(0, 6))
        self.x_box.bind("<<ComboboxSelected>>", lambda e: self.on_x_change())

        ttk.Label(side, text="Y columns (click to show on graph)").pack(anchor="w")
        self.y_list = self._listbox(side, "multiple")
        self.y_list.bind("<<ListboxSelect>>", lambda e: self.redraw())

        row = ttk.Frame(side)
        row.pack(fill="x", pady=(4, 2))
        ttk.Button(row, text="All", command=lambda: self.select_y(True)
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="None", command=lambda: self.select_y(False)
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(side, text="Move selected to working list  →",
                   command=self.move_to_working).pack(fill="x", pady=(2, 8))

        g = ttk.LabelFrame(side, text="Graph", padding=(8, 4))
        g.pack(fill="x", pady=(4, 0))
        g.columnconfigure(1, weight=1)
        g.columnconfigure(3, weight=1)

        def entry(var, row, col, span=1):
            e = ttk.Entry(g, textvariable=var, width=7)
            e.grid(row=row, column=col, columnspan=span, sticky="ew", pady=2)
            e.bind("<Return>", lambda ev: self.redraw())
            e.bind("<FocusOut>", lambda ev: self.redraw())

        def label(text, row):
            ttk.Label(g, text=text).grid(row=row, column=0, sticky="w", padx=(0, 6))

        self.title_var, self.xlabel_var, self.ylabel_var = (tk.StringVar() for _ in range(3))
        for row, (text, var) in enumerate((("Title", self.title_var),
                                           ("X label", self.xlabel_var),
                                           ("Y label", self.ylabel_var))):
            label(text, row)
            entry(var, row, 1, span=3)

        # axis ranges: min "to" max, blank = automatic
        self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var = (
            tk.StringVar() for _ in range(4))
        for row, text, lo, hi in ((3, "X range", self.xmin_var, self.xmax_var),
                                  (4, "Y range", self.ymin_var, self.ymax_var)):
            label(text, row)
            entry(lo, row, 1)
            ttk.Label(g, text="to").grid(row=row, column=2, padx=4)
            entry(hi, row, 3)
        ttk.Label(g, text="Blank = automatic. Put the larger X first to flip it.",
                  foreground="#666", font=("TkDefaultFont", 8)
                  ).grid(row=5, column=0, columnspan=4, sticky="w")

        label("Style", 6)
        self.style_var = tk.StringVar(value="Line")
        style_box = ttk.Combobox(g, state="readonly", values=list(STYLES),
                                 textvariable=self.style_var, width=7)
        style_box.grid(row=6, column=1, columnspan=3, sticky="ew", pady=2)
        style_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        self.log_var = tk.BooleanVar(value=False)
        self.grid_var = tk.BooleanVar(value=True)
        checks = ttk.Frame(g)
        checks.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(2, 2))
        ttk.Checkbutton(checks, text="Log Y", variable=self.log_var,
                        command=self.redraw).pack(side="left")
        ttk.Checkbutton(checks, text="Grid", variable=self.grid_var,
                        command=self.redraw).pack(side="left", padx=(8, 0))
        ttk.Button(checks, text="Reset axes", command=self.reset_axes
                   ).pack(side="right")

        ttk.Button(side, text="Save graph as image...",
                   command=self.save_image).pack(fill="x", pady=(6, 0))

    def _build_right_panel(self):
        side = ttk.Frame(self, padding=10, width=300)
        side.pack(side="right", fill="y")

        ttk.Label(side, text="Working list", font=("TkDefaultFont", 10, "bold")
                  ).pack(anchor="w")
        ttk.Label(side, text="Functions apply to the selected items,\n"
                             "or to all items if none are selected.",
                  foreground="#666").pack(anchor="w", pady=(0, 4))
        self.w_list = self._listbox(side, "extended", width=36)
        self.w_list.bind("<<ListboxSelect>>", lambda e: self._update_target_label())
        self.w_list.bind("<Double-Button-1>", lambda e: self.rename_item())

        row = ttk.Frame(side)
        row.pack(fill="x", pady=(4, 8))
        ttk.Button(row, text="← Y columns", command=self.move_back
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Rename", command=self.rename_item
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Deselect", command=self.deselect_working
                   ).pack(side="left", expand=True, fill="x")
        self.undo_btn = ttk.Button(row, text="Undo", command=self.undo,
                                   state="disabled")
        self.undo_btn.pack(side="left", expand=True, fill="x")

        ttk.Label(side, text="Function").pack(anchor="w")
        self.func_var = tk.StringVar(value=next(iter(FUNCTIONS)))
        func_box = ttk.Combobox(side, state="readonly", values=list(FUNCTIONS),
                                textvariable=self.func_var, height=20)
        func_box.pack(fill="x", pady=(0, 6))
        func_box.bind("<<ComboboxSelected>>", lambda e: self.on_function_change())

        self.value_label = ttk.Label(side, text="")
        self.value_label.pack(anchor="w")
        self.value_var = tk.StringVar()
        self.value_entry = ttk.Entry(side, textvariable=self.value_var)
        self.value_entry.pack(fill="x", pady=(0, 6))
        self.value_entry.bind("<Return>", lambda e: self.apply_function())

        self.apply_btn = ttk.Button(side, text="Apply", command=self.apply_function)
        self.apply_btn.pack(fill="x")
        self.target_label = ttk.Label(side, text="", foreground="#666")
        self.target_label.pack(anchor="w", pady=(2, 8))

        ttk.Label(side, text="Results").pack(anchor="w")
        res_frame = ttk.Frame(side)
        res_frame.pack(fill="x")
        self.results = tk.Text(res_frame, height=5, width=36, wrap="none",
                               font=("Consolas", 9))
        xscroll = ttk.Scrollbar(res_frame, orient="horizontal",
                                command=self.results.xview)
        self.results.config(xscrollcommand=xscroll.set)
        xscroll.pack(side="bottom", fill="x")
        self.results.pack(fill="x")
        ttk.Button(side, text="Clear results",
                   command=lambda: self.results.delete("1.0", "end")
                   ).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Export working list to Excel...",
                   command=self.export_working).pack(fill="x", pady=(4, 0))
        self.log_btn = ttk.Button(side, text="Export function log (CSV)...",
                                  command=self.export_log)
        self.log_btn.pack(fill="x", pady=(4, 0))

        self.on_function_change()

    def _listbox(self, parent, mode, width=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        lb = tk.Listbox(frame, selectmode=mode, exportselection=False, height=8,
                        **({"width": width} if width else {}))
        scroll = ttk.Scrollbar(frame, command=lb.yview)
        lb.config(yscrollcommand=scroll.set)
        lb.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return lb

    def _build_plot_area(self):
        area = ttk.Frame(self)
        area.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax_tauc = None                   # second graph, shown when there are Tauc plots
        self.ax.text(0.5, 0.5, "Open a file to begin", ha="center", va="center",
                     color="#888", fontsize=14, transform=self.ax.transAxes)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=area)
        # toolbar gives zoom, pan, reset and save
        self.toolbar = NavigationToolbar2Tk(self.canvas, area, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---- loading ----------------------------------------------------------
    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose a data file",
            filetypes=[("Spreadsheets", "*.xlsx *.xlsm *.xls *.csv *.tsv *.txt"),
                       ("All files", "*.*")])
        if path:
            self.open_file(path)

    def open_file(self, path):
        try:
            tables = load_tables(path)
        except Exception as err:
            messagebox.showerror("Could not open file", str(err))
            return
        if not tables:
            messagebox.showwarning(
                "Nothing to plot",
                "No sheet with at least two columns of data was found.")
            return

        self.tables = tables
        self.ws = WorkingSet()                # new file -> empty working list
        self.refresh_working()
        self.file_name = os.path.basename(path)
        self.file_label.config(text=self.file_name)
        for var in (self.title_var, self.xlabel_var, self.ylabel_var, self.xmin_var,
                    self.xmax_var, self.ymin_var, self.ymax_var):
            var.set("")
        self.log_event("Open file")
        names = list(tables)
        self.sheet_box.config(values=names)
        self.sheet_box.set(names[0])
        self.on_sheet_change()

    def on_sheet_change(self):
        self.df = self.tables[self.sheet_box.get()]
        cols = list(self.df.columns)
        self.x_box.config(values=cols)
        self.x_box.set(cols[0])               # first column = X by default
        self.on_x_change()

    def on_x_change(self):
        self.refresh_available(select_all=True)
        self.redraw()

    def key_for(self, col):
        """Working-list name for a column (sheet added if the file has several)."""
        return f"{col} ({self.sheet_box.get()})" if len(self.tables) > 1 else col

    # ---- the two lists ----------------------------------------------------
    def refresh_available(self, select_all=False, also_select=()):
        """Y columns of this sheet that aren't in the working list."""
        keep = {self.y_names[i] for i in self.y_list.curselection()} | set(also_select)
        x_col = self.x_box.get()
        self.y_names = [c for c in self.df.columns
                        if c != x_col and self.key_for(c) not in self.ws.sources()]
        self.y_list.delete(0, "end")
        for i, name in enumerate(self.y_names):
            self.y_list.insert("end", name)
            if select_all or name in keep:
                self.y_list.select_set(i)

    def refresh_working(self):
        keep = set(self.selected_work_keys())
        self.w_list.delete(0, "end")
        for i, key in enumerate(self.ws.keys()):
            self.w_list.insert("end", self.ws.label(key))
            if key in keep:
                self.w_list.select_set(i)
        self.undo_btn.config(state="normal" if self.ws.can_undo() else "disabled")
        self._update_target_label()

    def selected_work_keys(self):
        keys = self.ws.keys()
        return [keys[i] for i in self.w_list.curselection() if i < len(keys)]

    def select_y(self, on):
        if on:
            self.y_list.select_set(0, "end")
        else:
            self.y_list.select_clear(0, "end")
        self.redraw()

    def move_to_working(self):
        if self.df is None:
            return
        cols = [self.y_names[i] for i in self.y_list.curselection()]
        if not cols:
            messagebox.showinfo("Nothing selected",
                                "Click one or more Y columns on the left first.")
            return
        x = self.df[self.x_box.get()]
        self.ws.checkpoint()
        skipped = []
        for col in cols:
            try:
                key = self.ws.add(self.key_for(col), x, self.df[col])
                self.log_event("Move to working list", item=key)
            except ValueError:
                skipped.append(col)
        if skipped:
            messagebox.showwarning("Skipped", "No numeric data in: " + ", ".join(skipped))
        self.refresh_available()
        self.refresh_working()
        self.redraw()

    def move_back(self):
        """Send working-list items to the Y column list. An untouched column just
        goes back; anything changed or new (smoothed, averaged, fitted peaks...)
        is added as a NEW Y column, so the original stays as it was."""
        keys = self.selected_work_keys()
        if not keys:
            messagebox.showinfo("Nothing selected",
                                "Select items in the working list to send to the "
                                "Y columns.")
            return
        self.ws.checkpoint(extra=self._column_state())
        sheet_x = pd.to_numeric(self.df[self.x_box.get()], errors="coerce").to_numpy(float)
        added, failed, tauc = [], [], []
        for k in keys:
            it = self.ws.items[k]
            if it.get("tauc"):
                tauc.append(k)
                continue
            if it.get("source") and not it["ops"] and not it.get("renamed"):
                self.log_event("Move to Y columns", item=k, result="original column")
                continue
            # put the item's values on this sheet's X points (blank outside its range)
            y = np.interp(sheet_x, it["x"], it["y"], left=np.nan, right=np.nan)
            y[np.isnan(sheet_x)] = np.nan
            if np.isnan(y).all():
                failed.append(k)
                continue
            name = self._unique_column(self.ws.label(k).replace("  [", " ["))
            self.df[name] = y
            added.append(name)
            self.log_event("Move to Y columns", item=k, result=f"new column '{name}'")
        self.ws.remove([k for k in keys if k not in failed and k not in tauc])
        if tauc:
            messagebox.showwarning("Not moved",
                                   "Tauc plot items have photon energy as X, so they "
                                   "can't go onto the sheet. Use Export instead:\n"
                                   + "\n".join(tauc))
        if failed:
            messagebox.showwarning("Not moved",
                                   "These don't overlap this sheet's X values, so they "
                                   "stayed in the working list:\n" + "\n".join(failed))
        self.w_list.selection_clear(0, "end")
        self.refresh_available(also_select=added)
        self.refresh_working()
        self.redraw()

    def _unique_column(self, base):
        name, n = base, 2
        while name in self.df.columns:
            name, n = f"{base} #{n}", n + 1
        return name

    def _column_state(self):
        """Column lists of every sheet, so Undo can remove columns added later."""
        return {sheet: list(df.columns) for sheet, df in self.tables.items()}

    def rename_item(self):
        keys = self.selected_work_keys()
        if len(keys) != 1:
            messagebox.showinfo("Rename",
                                "Select one item in the working list to rename\n"
                                "(or double-click it).")
            return
        old = keys[0]
        new = simpledialog.askstring("Rename", "New name (shown in the legend):",
                                     initialvalue=self.ws.label(old).replace("  [", " ["),
                                     parent=self)
        if new is None or (new.strip() == old and not self.ws.items[old]["ops"]):
            return
        self.ws.checkpoint()
        try:
            self.ws.rename(old, new)
        except ValueError as err:
            self.ws.undo()
            messagebox.showerror("Could not rename", str(err))
            return
        self.log_event("Rename", item=old, result=new.strip())
        self.refresh_working()
        idx = self.ws.keys().index(new.strip())
        self.w_list.selection_clear(0, "end")
        self.w_list.select_set(idx)
        self._update_target_label()
        self.redraw()

    def deselect_working(self):
        self.w_list.selection_clear(0, "end")
        self._update_target_label()

    def undo(self):
        columns = self.ws.undo()
        if columns:                           # remove Y columns added since then
            for sheet, cols in columns.items():
                if sheet in self.tables:
                    self.tables[sheet] = self.tables[sheet][cols]
            self.df = self.tables[self.sheet_box.get()]
            if self.x_box.get() not in self.df.columns:
                self.on_sheet_change()
        self.w_list.selection_clear(0, "end")
        self.log_event("Undo")
        self.refresh_available()
        self.refresh_working()
        self.redraw()

    # ---- functions --------------------------------------------------------
    def on_function_change(self):
        spec = FUNCTIONS[self.func_var.get()]
        if spec["hint"]:
            self.value_label.config(text=spec["hint"])
            self.value_entry.config(state="normal")
            self.value_var.set(spec["default"])
        else:
            self.value_label.config(text="(no value needed)")
            self.value_var.set("")
            self.value_entry.config(state="disabled")
        self._update_target_label()

    def _update_target_label(self):
        n_sel, n_all = len(self.selected_work_keys()), len(self.ws.items)
        if n_all == 0:
            text = "Working list is empty."
        elif n_sel:
            text = f"Will apply to {n_sel} selected item(s)."
        else:
            text = f"Will apply to all {n_all} item(s)."
        if self.func_var.get() == "Average" and n_all:
            text += "\nUses the X points of the top item."
        if self.func_var.get().startswith("Tauc") and n_all:
            text += "\nX must be wavelength in nm."
        self.target_label.config(text=text)

    def apply_function(self):
        name = self.func_var.get()
        keys = self.selected_work_keys() or self.ws.keys()
        try:
            report = self.ws.apply(name, keys, self.value_var.get())
        except Exception as err:
            messagebox.showerror("Could not apply", str(err))
            return
        value = self.value_var.get().strip()
        if self.ws.details:
            for d in self.ws.details:
                self.log_event(name, value=value, **d)
        else:
            for k in keys:
                self.log_event(name, item=k, value=value,
                               result=self.ws.label(k) if k in self.ws.items else "")
        if report:
            header = name + (f" ({value})" if value else "")
            self.results.insert("end", f"{header}\n{report}\n\n")
            self.results.see("end")
        self.refresh_working()
        self.redraw()

    # ---- drawing and saving -----------------------------------------------
    def redraw(self):
        if self.df is None:
            return
        x_col = self.x_box.get()
        series = [(c, self.df[x_col], self.df[c], "original")
                  for c in (self.y_names[i] for i in self.y_list.curselection())]
        work = [(self.ws.label(k), it["x"], it["y"],
                 "fit" if it.get("dashed") else "working", it.get("tauc"))
                for k, it in self.ws.items.items()]
        series += [w[:4] for w in work if not w[4]]
        tauc = [w for w in work if w[4]]
        self._set_panels(bool(tauc))
        self.ax.set_axis_on()
        self.ax.set_yscale("linear")
        draw_plot(self.ax, series,
                  style=self.style_var.get(), log_y=self.log_var.get(),
                  grid=self.grid_var.get(), title=self.title_var.get().strip(),
                  x_label=self.xlabel_var.get().strip() or x_col,
                  y_label=self.ylabel_var.get().strip(),
                  xlim=(self._axis_value(self.xmin_var), self._axis_value(self.xmax_var)),
                  ylim=(self._axis_value(self.ymin_var), self._axis_value(self.ymax_var)))
        if tauc:
            kinds = {w[4] for w in tauc}
            y_label = (r"$(\alpha h\nu)^2$" if kinds == {"direct"} else
                       r"$(\alpha h\nu)^{1/2}$" if kinds == {"indirect"} else
                       r"$(\alpha h\nu)^{1/n}$")
            draw_plot(self.ax_tauc, [w[:4] for w in tauc],
                      style=self.style_var.get(), grid=self.grid_var.get(),
                      title="Tauc plot", x_label="Photon energy (eV)", y_label=y_label)
            self.ax_tauc.axhline(0, color="#888", linewidth=0.8)
        self.canvas.draw_idle()

    def _set_panels(self, tauc):
        """One graph, or two stacked when there are Tauc plots (spectra on
        top, Tauc plot below). The axis range boxes apply to the top graph."""
        if tauc == (self.ax_tauc is not None):
            return
        self.fig.clear()
        if tauc:
            self.ax, self.ax_tauc = self.fig.subplots(2, 1)
        else:
            self.ax, self.ax_tauc = self.fig.add_subplot(111), None
        self.toolbar.update()                 # reset zoom history for the new graphs

    @staticmethod
    def _axis_value(var):
        """Number typed in an axis box, or None (automatic) if blank/not a number."""
        try:
            return float(var.get().strip())
        except ValueError:
            return None

    def reset_axes(self):
        for var in (self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var):
            var.set("")
        self.redraw()

    def save_image(self):
        if self.df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            self.fig.savefig(path, dpi=300)

    def export_working(self):
        if not self.ws.items:
            messagebox.showinfo("Nothing to export", "The working list is empty.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        table = self.ws.to_frame()
        try:
            if path.lower().endswith(".csv"):
                table.to_csv(path, index=False)
            else:
                table.to_excel(path, index=False)
        except Exception as err:
            messagebox.showerror("Could not save", str(err))


    # ---- function log -----------------------------------------------------
    def log_event(self, action, **fields):
        """Add a row to the function log. fields: item, value, result, peak,
        center, height, fwhm, area, baseline, r2."""
        from datetime import datetime
        row = dict.fromkeys(LOG_COLUMNS, "")
        row.update(Time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   File=self.file_name, Action=action)
        names = {"fwhm": "FWHM", "r2": "R2"}
        for key, val in fields.items():
            col = names.get(key, key.capitalize())
            if isinstance(val, (float, np.floating)):
                val = float(f"{val:.6g}")
            row[col] = val
        self.log.append(row)
        self.log_btn.config(text=f"Export function log (CSV)...  ({len(self.log)})")

    def export_log(self):
        if not self.log:
            messagebox.showinfo("Nothing to export", "No functions have been applied yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="function_log.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            pd.DataFrame(self.log, columns=LOG_COLUMNS).to_csv(path, index=False)
        except Exception as err:
            messagebox.showerror("Could not save", str(err))


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
