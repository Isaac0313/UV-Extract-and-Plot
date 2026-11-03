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
A separate, standalone program: one electron moving between two nuclei (an H2+-like molecule).
The electron can be a **classical particle** (pushed around by the Coulomb attraction of both nuclei)
or a **wavefunction** (time-dependent Schrödinger equation in the same potential), and you can switch
between the two at any moment.

## What you can do
- **Become wave ψ**: the particle turns into a Gaussian wave packet with the same position and momentum.
- **Measure (collapse)**: the wave turns back into a particle; position drawn from |ψ(r)|², momentum from |ψ(k)|².
- **Relax → σ / σ\***: the wave settles (imaginary-time evolution) into the bonding or antibonding molecular orbital.
- **Auto switch**: random measure / become-wave events at a chosen rate.
- **Classical ghost**: a classical electron runs alongside the wave packet so you can compare the two motions.
- **Classical cloud**: the time-averaged position of the particle, to compare with |ψ|² of the orbitals.
- Sliders for bond length R, nuclear charges Z₁ and Z₂, softening a, wave packet width and speed.
- Click the plot to place the electron, drag to give it a velocity.

## Dependencies
- numpy, matplotlib, tkinter (`python -m pip install numpy matplotlib`; on Linux also `sudo apt install python3-tk`)

## Running
```bash
python MO-Electron-Simulator.py
```
```bash
python MO-Electron-Simulator.py --selftest
```
`--selftest` runs physics checks without opening a window (energy and norm conservation, orbital symmetry, measurement statistics).

## Model assumptions
- Atomic units (bohr, hartree, a.u. of time = 0.0242 fs).
- 2D: the electron moves in the plane that contains both nuclei.
- Soft-core Coulomb potential V = −Z / √(d² + a²) so the grid and Newton's equations stay well behaved.
- Nuclei are fixed (Born–Oppenheimer). Because of the 2D geometry and softening, energies differ from real 3D H2+.
