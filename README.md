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
- Sliders for bond length R, nuclear charges Z₁ and Z₂, softening a, wave packet width and speed.
- Click the plot to place an electron, drag to give it a velocity.

## How two electrons interact
- particle + particle: two point charges that repel each other.
- particle + wave: the particle feels the wave's charge cloud; the wave feels the particle's point charge.
- wave + wave: each wave feels the other's charge cloud (time-dependent Hartree / mean field). This keeps the cost at two 2D grids instead of one 4D grid, but leaves out electron correlation. The electrons are taken to have opposite spins, so both may occupy σ.

## Dependencies
- numpy, matplotlib, tkinter (`python -m pip install numpy matplotlib`; on Linux also `sudo apt install python3-tk`)

## Running
```bash
python MO-Electron-Simulator.py
```
```bash
python MO-Electron-Simulator.py --selftest
```
`--selftest` runs physics checks without opening a window (energy and norm conservation for one and two electrons, orbital symmetry, measurement statistics, the two-electron ground state).

## Model assumptions
- Atomic units (bohr, hartree, a.u. of time = 0.0242 fs).
- 2D: the electrons move in the plane that contains both nuclei.
- Soft-core Coulomb interactions 1 / √(d² + a²) (electron–nucleus and electron–electron) so the grid and Newton's equations stay well behaved.
- Nuclei are fixed (Born–Oppenheimer). Because of the 2D geometry and softening, energies differ from real 3D H2+ / H2.
