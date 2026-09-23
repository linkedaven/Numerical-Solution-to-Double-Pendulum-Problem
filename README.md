# Double Pendulum Visualizer

An interactive, real-time simulation and visualization of a **double pendulum** — a classic chaotic dynamical system. Built with `matplotlib` and `scipy`, with optional JIT acceleration via `numba`.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Live animated pendulum** with trailing motion path
- **Angular motion plot** (θ₁, θ₂ vs. time)
- **Phase-space trajectories** for both bob masses
- **Interactive controls** — tweak initial angles, angular velocities, masses, rod lengths, simulation time span, resolution, and playback speed, then hit **Run**
- **Play/Pause** toggle for the animation
- Dark-themed UI
- Optional `numba` JIT compilation for faster integration on longer runs

## How it works

The equations of motion are derived analytically using **Lagrangian mechanics**: starting from the kinetic and potential energy of the two bobs, the Lagrangian $L = T - V$ is formed, and applying the Euler-Lagrange equation to each generalized coordinate ($\theta_1$, $\theta_2$) yields a pair of coupled, nonlinear second-order ODEs for the angular accelerations.

These equations have no closed-form solution, so they're solved **numerically** with `scipy.integrate.solve_ivp` (DOP853, an 8th-order Runge-Kutta method) for high accuracy — this system is chaotic, so small integration errors compound quickly, and low-order or fixed-step methods drift noticeably over long runs. `matplotlib.widgets` provides the text boxes and buttons, and `FuncAnimation` drives the live rendering.

## Installation

```bash
git clone https://github.com/<your-username>/double-pendulum-vis.git
cd double-pendulum-vis
pip install -r requirements.txt
```

## Usage

```bash
python dpend_vis.py
```

A window will open with the simulation running using default parameters (θ₁=120°, θ₂=20°, m₁=m₂=2 kg, l₁=l₂=1 m). Edit any of the input boxes and click **Run** to re-simulate with new parameters, or **Pause**/**Play** to control the animation.

> **Tip:** Install `numba` (`pip install numba`) for noticeably faster simulation, especially with a large number of points.

## Requirements

- Python 3.9+
- numpy
- scipy
- matplotlib
- numba (optional, recommended)

## License

MIT — see [LICENSE](LICENSE).