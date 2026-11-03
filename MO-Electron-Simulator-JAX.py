"""
MO Electron Simulator - JAX version
-----------------------------------
The same window as MO-Electron-Simulator.py, with the physics running on
JAX (mo_physics_jax.py): each animation frame is one compiled loop, on the
CPU or - with a CUDA build of JAX - on an NVIDIA GPU. See
MO-Electron-Simulator.py for what the buttons and sliders do.

Files needed in the same folder: MO-Electron-Simulator.py, mo_physics.py,
mo_physics_jax.py, mo_controller.py

Setup (once):
    python -m pip install numpy matplotlib jax
    (NVIDIA GPU: python -m pip install "jax[cuda12]")

Run:
    python MO-Electron-Simulator-JAX.py
    python MO-Electron-Simulator-JAX.py --fp32        (single precision: ~2x faster FFTs;
                                                       use this on a consumer NVIDIA GPU)
    python MO-Electron-Simulator-JAX.py --selftest    (physics checks on JAX, no window)

The first time each combination of settings is used (1 or 2 electrons,
particle or wave, vibrating nuclei, ...) JAX compiles it, which pauses the
animation for a second or two; after that it runs at full speed.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.argv.append("--jax")
    runpy.run_path(os.path.join(here, "MO-Electron-Simulator.py"), run_name="__main__")
