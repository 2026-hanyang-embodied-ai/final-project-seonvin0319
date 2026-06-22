# PathBridger — Goal-Conditioned Control via Forward-Bridge Planning + Inverse Dynamics

PathBridger reaches far-away goals in offline goal-conditioned control by chaining three
learned components instead of a single reactive policy:

1. **Subgoal prediction** — predict a reachable intermediate endpoint $\hat{s}_{t+K}$ from the
   current state $s_t$ and the goal $g$.
2. **Forward-bridge planning** — connect $s_t$ to $\hat{s}_{t+K}$ with a closed-form linear-SDE
   Gaussian *bridge* plus a learned endpoint-preserving residual, giving a smooth state plan
   $z_0, z_1, \dots, z_K$.
3. **Inverse dynamics (IDM)** — decode actions from consecutive planned states
   $(z_i, z_{i+1})$, execute a short action chunk, then re-plan.

This repository is the **course deliverable**: a minimal, self-contained slice of the project
that runs the **IDM control path** end-to-end on `antmaze-large-navigate-v0`. The full training
pipeline (which also trains a transitive value critic and an SPI actor) is part of the larger
PathBridger research codebase; here we keep only what is needed to *run* the IDM path.

## Links

| Item | Link |
|------|------|
| Presentation video | https://www.youtube.com/watch?v=5PU-iD3_U5w |
| Presentation slides | https://drive.google.com/file/d/1NA7Dv9fKVhcGJRXG9ldH9bQMRTZwykkp/view?usp=drive_link |
| Report | https://drive.google.com/file/d/1w49Atxus9OucPWTQ2wb5WTKNdWf0U-C3/view?usp=drive_link |
| Dataset | https://drive.google.com/drive/folders/1A3vJx4yImA0S7pFlx9TY76koLxSardWO?usp=drive_link |
| Demo video | https://drive.google.com/drive/folders/1VMgUiczm3jaFYt94v6I5tH4P0rQdGy3d?usp=drive_link |

## How to run

```bash
# 1. Create an environment (Python 3.10, CUDA 12 GPU recommended)
pip install -r requirements.txt

# 2. (optional) register the kernel used by the notebook
python -m ipykernel install --user --name offrl --display-name "Python (offrl)"

# 3. Open and run the notebook top-to-bottom
jupyter notebook final-project.ipynb
```

`final-project.ipynb` is already executed and contains saved outputs. It walks through:

1. Setup and imports
2. The hard-coded AntMaze-Large config (matches the trained checkpoint)
3. Building the OGBench env + offline dataset
4. A short **training demo** of the standalone dynamics agent (loss decreases)
5. Loading the **600-epoch checkpoint**
6. **Evaluation** — IDM rollout success rate over the 5 eval tasks (overall ≈ 0.84)
7. **Rendering** an IDM rollout video, displayed inline
8. Notes on running other environments

The OGBench dataset is downloaded/cached automatically on first run; a copy is also linked
above.

## Running other environments

The notebook hard-codes the AntMaze-Large config, but the IDM path is environment-agnostic. To
target another OGBench task (`antmaze-medium-navigate`, `cube-double-play`, `cube-triple-play`,
`puzzle-3x3-play`, `puzzle-4x4-play`, …), change `ENV_NAME`, swap in that environment's config
overrides, and point the checkpoint path at the matching `params_<epoch>.pkl`. Maze-navigation
and manipulation-play environments are dispatched automatically by the rollout helpers. See
section 8 of the notebook for details.

## Repository layout

```
.
├── final-project.ipynb        # main deliverable (executed)
├── README.md
├── requirements.txt
├── pb/                        # minimal PathBridger modules needed for the IDM path
│   ├── agents/                #   dynamics agent (+ critic value-net definition)
│   ├── utils/                 #   datasets, env utils, bridge math, networks, IO
│   └── rollout/               #   chunked episode runner, env sync, rendering
├── checkpoint/
│   ├── dynamics/params_600.pkl  # best AntMaze-Large dynamics checkpoint (600 epochs)
│   └── flags.json               # resolved config of that run (reference)
└── renders/
    └── idm_task1.mp4          # example IDM rollout video
```

## Notes

- A CUDA-12 GPU is recommended; the notebook uses JAX with the `cuda12` backend.
- Rendering uses headless MuJoCo via EGL (`MUJOCO_GL=egl`).
