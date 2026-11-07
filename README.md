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

# Opening in VS Code
1. *File → Open Folder* and pick this folder. Accept the suggested Python extension.
2. *Ctrl+Shift+P → Python: Create Environment → Venv*, and tick `requirements.txt` so everything is installed. Or, in VS Code's terminal: `python -m pip install -r requirements.txt`.
3. Open *Run and Debug* (Ctrl+Shift+D), pick a program from the list (simulators, neural-network runs, self-tests, UV-Vis plotter) and press F5.

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

# Neural-network wavefunction solver (`mo_nn_solver.py`)
Finds the ground state of the simulator's molecule (one or two electrons, same 2D soft-core model) with a neural network as the wavefunction, trained by variational Monte Carlo in JAX: the method behind FermiNet / PauliNet, sized for this model.

- **Wavefunction**: ψ(r₁, r₂) = exp(f), f = exponential envelope around the nuclei + a neural network. It is built so that swapping the electrons leaves ψ unchanged (spin singlet), which is what the ground state of two opposite-spin electrons needs.
- **Energy**: the average of the local energy E_L = Hψ/ψ over electron positions sampled from |ψ|² (Metropolis). JAX computes the derivatives in E_L exactly.
- **Training**: Adam on the energy gradient 2⟨(E_L − E) ∂ log ψ⟩. The variational principle keeps the energy above the true ground state, and E_L becomes constant (zero variance) for the exact state, so the variance shows how good the network is.
- **Answer keys**: one electron is checked against the simulator's grid solver; two electrons against mean field (what the simulator's "Relax both" gives, which misses electron correlation) and against an exact solution on a full 4D grid (x₁, y₁, x₂, y₂), which is affordable only because this model is 2D with two electrons.

```bash
python mo_nn_solver.py
```
```bash
python mo_nn_solver.py --electrons 1
```
```bash
python mo_nn_solver.py --curve 1.0,1.5,2.0,3.0,4.0
```
```bash
python mo_nn_solver.py --selftest
```
Options: `--R`, `--Z1`, `--Z2`, `--soft`, `--iterations`, `--walkers`, `--hidden`, `--seed`, `--out DIR` (plots, default `nn_results/`), `--no-exact` (skip the 4D grid), `--fp32` (single precision; use on GPUs).

Results on a 4-core CPU (R = 2 bohr, default settings):

| Method | H2 energy (Ha) |
|---|---|
| Mean field (shared orbital) | −1.880275 |
| Neural network (3000 iterations, ~2 min) | −1.932252 ± 0.000019 |
| Exact, 4D grid (dx 0.30 bohr) | −1.932287 |

The network recovers 99.9% of the correlation energy (exact − mean field = 0.052 Ha). For H2+ it matches the grid solver to within 0.0001 Ha after 600 iterations.

Energy vs bond length (`--curve 1.0,1.5,2.0,3.0,4.0`, total energies including nuclear repulsion, about 20 minutes on a 4-core CPU):

| R (bohr) | Neural network | Exact (4D grid) | Mean field |
|---|---|---|---|
| 1.0 | −1.32165 ± 0.00003 | −1.32164 | −1.28879 |
| 1.5 | −1.44566 ± 0.00003 | −1.44566 | −1.40507 |
| 2.0 | −1.43229 ± 0.00003 | −1.43229 | −1.38028 |
| 3.0 | −1.35676 ± 0.00002 | −1.35680 | −1.26653 |
| 4.0 | −1.32046 ± 0.00002 | −1.32049 | −1.17497 |

Mean field gets worse as the bond stretches (0.15 Ha too high at R = 4: it forces both electrons into one shared orbital, while really each electron goes to its own atom). The network stays within 0.00004 Ha of the exact answer at every bond length. The exact grid values carry their own discretization error of about 0.00002 Ha.

