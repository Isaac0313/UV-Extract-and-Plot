# UV-Extract-and-Plot (WIP)
## About me / Goal
- I have ran a lot of UV-vis and photoluminescence spectroscopy within my undergraduate and graduate time. The most frustrating part for me is making each one of the plots, grouping them, making averages etc. The hope is that this software makes it easier for you to do this. 
## Dependencies Installed
- copy
- os
- sys
- tkinter imported -> ttk, filedialog, messagebox, simpledialog, colorchooser
- ctypes imported -> windll
- numpy
- pandas
- matplotlib.figure imported -> Figure
- matplotlib.backends.backend_tkagg imported -> FigureCanvasTkAgg, NavigationToolbar2T

## Set-Up Guide
- This software is all under one .py code. This means running the .py will open up the interface.
## Main Use
- Averaging groups of data set to determine overall prefomance of a type of material.

# MO Electron Simulator (`MO-Electron-Simulator.py`)
A separate, standalone program: one or two electrons moving between two nuclei (an H2+ / H2-like molecule).
Each electron can be a **classical particle** (pushed around by the Coulomb forces of the nuclei and the other electron)
or a **wavefunction** (time-dependent Schrödinger equation in the same potential), and you can switch any electron
between the two at any moment. The question it answers: *given a starting position, does the electron stay in the orbital?*

## What you can do
- **Number of electrons: 1 or 2.** A second electron starts at the mirror image of the first; "Act on" picks which electron the buttons change.
- **Become wave ψ**: a particle turns into a Gaussian wave packet with the same position and momentum.
- **Measure (collapse)**: a wave turns back into a particle; position drawn from |ψ(r)|², momentum from |ψ(k)|².
- **Relax → σ / σ\***: the wave settles (imaginary-time evolution) into the bonding or antibonding orbital. With two electrons and "both", they settle into one shared σ orbital.
- **Inside-the-orbital tracking**: the dashed line encloses 90% of the σ orbital; a plot shows, for each electron, whether it is inside (particle) or how much of it is inside (wave), and the status bar gives the time average.
- **Auto switch**: random measure / become-wave events at a chosen rate.
- **Classical ghost** (one electron): a classical electron runs alongside the wave packet so you can compare the two motions.
- **Classical cloud**: the time-averaged position of a particle, to compare with |ψ|².
- **Nuclei vibrate**: the nuclei move along the bond, pushed apart by their repulsion and pulled together by the electrons (point charges or charge clouds). A plot shows the bond length R(t). The nuclear-mass slider (hydrogen = 1836 electron masses) can make them lighter so the vibration (~10 fs for real H) is quicker to watch.
- Sliders for bond length R, nuclear charges Z₁ and Z₂, softening a, nuclear mass, wave packet width and speed.
- Click the plot to place an electron, drag to give it a velocity.

## How two electrons interact
- particle + particle: two point charges that repel each other.
- particle + wave: the particle feels the wave's charge cloud; the wave feels the particle's point charge.
- wave + wave: each wave feels the other's charge cloud (time-dependent Hartree / mean field). This keeps the cost at two 2D grids instead of one 4D grid, but leaves out electron correlation. The electrons are taken to have opposite spins, so both may occupy σ.

## Three versions
All three have the same buttons, sliders and physics; keep all the files below in one folder.

| Program | Window | Physics | Install |
|---|---|---|---|
| `MO-Electron-Simulator.py` | tkinter + matplotlib | NumPy | `pip install numpy matplotlib` (+ `python3-tk` on Linux) |
| `MO-Electron-Simulator-JAX.py` | tkinter + matplotlib | JAX (CPU, or NVIDIA GPU) | also `pip install jax` (GPU: `pip install "jax[cuda12]"`) |
| `MO-Electron-Simulator-Qt.py` | pyqtgraph (Qt) | NumPy, or JAX with `--jax` | `pip install numpy pyqtgraph PyQt6` (+ `jax` for `--jax`) |

Shared modules: `mo_physics.py` (NumPy physics + self-test), `mo_physics_jax.py` (the same physics compiled with JAX), `mo_controller.py` (what the buttons/sliders do, used by both windows).

```bash
python MO-Electron-Simulator.py
```
```bash
python MO-Electron-Simulator-JAX.py
```
```bash
python MO-Electron-Simulator-Qt.py --jax --fp32
```
- `--jax` (tkinter and Qt programs): use the JAX physics.
- `--fp32`: run JAX in single precision. About 2x faster FFTs on a CPU, and the right choice on consumer NVIDIA GPUs, whose 64-bit maths is much slower. Results agree with 64-bit to the digits shown.
- The first time a combination of settings is used (1 or 2 electrons, particle or wave, vibrating nuclei, ...), JAX compiles it: a pause of a second or two.

Measured frame rates on a 4-core CPU without a GPU (higher is better):

| Case | tkinter, NumPy | pyqtgraph, NumPy | pyqtgraph, JAX 32-bit |
|---|---|---|---|
| 1 particle | ~45 fps | ~63 fps | ~61 fps |
| 1 wave | ~34 fps | ~30 fps | ~41 fps |
| 2 waves | ~14 fps | ~16 fps | ~25 fps |
| 2 waves + vibrating nuclei | ~7 fps (10 steps/frame) | ~11 fps | ~20 fps |

On a CPU the wave cases are limited by the FFTs; a GPU with JAX should help most there.

## Self-test
```bash
python MO-Electron-Simulator.py --selftest
```
```bash
python MO-Electron-Simulator-JAX.py --selftest
```
`--selftest` runs physics checks without opening a window (energy and norm conservation for one and two electrons, orbital symmetry, measurement statistics, the two-electron ground state, forces on the nuclei, energy conservation with vibrating nuclei).

## Model assumptions
- Atomic units (bohr, hartree, a.u. of time = 0.0242 fs).
- 2D: the electrons move in the plane that contains both nuclei.
- Soft-core Coulomb interactions 1 / √(d² + a²) (electron–nucleus and electron–electron) so the grid and Newton's equations stay well behaved.
- Nuclei are fixed (Born–Oppenheimer) unless "Nuclei vibrate" is on; then they are classical point masses moving along the bond axis (no rotation).
- Because of the 2D geometry and softening, energies and bond lengths differ from real 3D H2+ / H2.
