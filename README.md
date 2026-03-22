# FyrdUp — uAgents, Google ADK & Gemini, ROS 2 & Gazebo Harmonic

Multi-agent wildfire simulation + framework for **BeachHacks 2026**: a **scout** coordinates **N firefighter** ground robots using **Fetch uAgents**, a **Google ADK** scout agent, and **Gemini** for vision-driven fire initialization and tactical assignment. Physics and terrain run in **Gazebo Harmonic**; fire spread is simulated on a ROS 2 grid and visualized in **Foxglove Studio** via embedded **foxglove-sdk** WebSocket servers.

![Reference aerial image used for terrain & fire initialization](wildfire.png)

## Overview

| Layer | Role |
|-------|------|
| **Gazebo Harmonic** | Headless world (`gz sim -s -r`), local install on the host |
| **ROS 2 Humble** | Topics for grid, odometry, cmd_vel, water, bridge |
| **Fetch uAgents** | Scout + firefighter messaging (`MoveCommand`, `RefillCommand`, alerts) |
| **Google ADK + Gemini** | `ScoutADKAgent`: **fast** streaming commentary + **slow** tactical cycle (deterministic or JSON assignments); slow loop may reference the live reasoning text |
| **VLM + depth** | Gemini labels fire shape from `wildfire.png`; **Depth Anything V2** builds a heightmap (PyTorch) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│     Gazebo Harmonic (headless: gz sim -s -r, local host)    │
└─────────────────────────────────────────────────────────────┘
                              │
                       ROS 2 Humble
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
fire_grid_node           scout (ROS)          firefighter_1..N
  /fire_grid            position_controller     navigation + water
  /water_spray          /scout/...              /cmd_vel, /water_level
     │                        │                        │
     └────────────────────────┼────────────────────────┘
                              │
              foxglove_viz (8765) + scene_publisher_3d (8766)
                    foxglove-sdk WebSocket servers
                              │
                     Foxglove Studio (2 connections)
                              │
                        ros_bridge node
                              │
              ┌───────────────┴───────────────┐
         ScoutUAgent (8000)              FirefighterAgent (8001+)
         + ScoutADKAgent                  uAgents
```

Visualization does **not** use `ros-humble-foxglove-bridge`. Two **foxglove-sdk** servers run inside the workspace:

| Port | Node | Purpose |
|------|------|---------|
| **8765** | `foxglove_viz` | Bird’s-eye `/viz/fire_image`, stats, agent status |
| **8766** | `scene_publisher_3d` | 3D `SceneUpdate` on `/scene`, `FrameTransform` on `/tf`, typed **`foxglove.Log`** on **`/scout_fast_reasoning`** (fast loop) and **`/scout_slow_reasoning`** (slow loop) |

Open **two** Foxglove WebSocket connections (`ws://localhost:8765` and `ws://localhost:8766`) to see 2D and 3D together. For the 3D panel, set **Fixed frame** to `world`.

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| ROS 2 | **Humble** | Native install or [RoboStack](https://robostack.github.io/) conda |
| Gazebo | **Harmonic** (`gz` 8.x) | **Host install** — not inside Docker/conda; [macOS](https://gazebosim.org/docs/harmonic/install/) via Homebrew `gz-harmonic` |
| Python | 3.10+ (3.11–3.12 typical with conda) | Must match what `colcon` passes to CMake |
| Foxglove Studio | [Latest](https://foxglove.dev/download) | Desktop or web — connects to localhost WebSockets |
| Google API key | — | `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini (fire init + scout) |

### Gazebo Harmonic (host)

```bash
# macOS (Apple Silicon): deactivate conda first — avoids lib conflicts with Homebrew
conda deactivate
brew tap osrf/simulation && brew install gz-harmonic
gz sim --version   # expect 8.x.x
```

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install gz-harmonic
gz sim --version
```

### RoboStack (optional, recommended on macOS)

```bash
conda create -n wildfire python=3.11
conda activate wildfire
conda install -c robostack-staging ros-humble-desktop ros-humble-ros-gz
```

You do **not** need `ros-humble-foxglove-bridge` for this repo’s visualization.

## Quick setup

From the repository root (with ROS 2 already sourced):

```bash
./setup.sh
source install/setup.bash
```

`setup.sh` installs Python deps, runs `colcon build` (with macOS-friendly CMake Python flags), and creates `install/<pkg>/lib/<pkg>` symlinks so `ros2 launch` finds Python entry points.

### Manual setup (same as above without the script)

On **macOS**, Homebrew’s default `python3` is [PEP 668](https://peps.python.org/pep-0668/) “externally managed,” so plain `pip install` fails. Activate your **conda ROS env** (e.g. `conda activate ros2`) or a **venv** before installing.

```bash
# macOS: conda (matches run_sim.sh) or: python3 -m venv .venv && source .venv/bin/activate
conda activate ros2

pip install -r requirements.txt

# Linux
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=$(which python3)

# macOS (often omit --symlink-install; see setup.sh)
colcon build --cmake-args -DPython3_EXECUTABLE=$(which python3) -DPython_EXECUTABLE=$(which python3)

# macOS: symlink executables into lib/<pkg>/
for pkg in scout_robot firefighter_robot wildfire_agents; do
  mkdir -p "install/$pkg/lib/$pkg"
  for script in install/$pkg/bin/*; do
    [ -f "$script" ] && ln -sf "$(pwd)/$script" "install/$pkg/lib/$pkg/$(basename "$script")"
  done
done

source install/setup.bash
```

### Python dependencies note

`requirements.txt` pins **uagents 0.24+** and **google-adk** (both use **pydantic v2**). First run also downloads **Depth Anything V2** weights via Hugging Face — expect a large download and GPU/CPU time.

### Optional: `run_sim.sh` (developer convenience)

`run_sim.sh` is a **macOS-oriented** helper that kills stale processes, activates a specific conda env, rebuilds, sources `.env` if present, and launches the sim. Adjust paths inside the script for your machine or use the generic `ros2 launch` flow below.

## Configuration & environment

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini (fire shape from image, scout assignments) |
| `WILDFIRE_IMAGE_PATH` | Aerial image for Depth Anything + VLM fire mask (default: `wildfire.png` in repo root) |
| `NUM_FIREFIGHTERS` | Set by launch; number of firefighter uAgents (`8001` …) |
| `SCOUT_UAGENT_MAILBOX` | `true` (default): scout registers with **Agentverse mailbox** for visibility; `false` for local-only scout (no mailbox polling) |
| `FIREFIGHTER_UAGENT_MAILBOX` | `true` (default): each firefighter uAgent uses **Agentverse mailbox**; `false` for local-only firefighters |
| `USE_VLM` | Passed through to the scout stack (`use_vlm` launch arg) |
| `ADK_MODEL` | Override Gemini model for the scout tactical / JSON step (default `gemini-2.5-flash-lite`) |
| `REASONING_MODEL` | Gemini model for the **fast** streaming commentary loop (default `gemini-2.5-flash-lite`) |
| `REASONING_INTERVAL` | Seconds between fast-loop Gemini calls (default from `agent_config.yaml`, else `5`) |
| `REASONING_ENABLED` | `true` / `false` — disable the fast loop for offline demos (default `true`) |
| `ANALYSIS_INTERVAL` | Seconds between slow `run_analysis` cycles (default from `agent_config.yaml` `analysis_interval`) |
| `SLOW_LOOP_REASONING_WINDOW_SEC` | How many seconds of fast-loop commentary the **slow** tactical Gemini call may see (default `10`) |

You can place keys in a **`.env`** file in the repo root and `source` it before launch (do **not** commit secrets).

## Running

```bash
export GOOGLE_API_KEY="your-key-here"
# optional: export WILDFIRE_IMAGE_PATH="/path/to/aerial.png"

source install/setup.bash
ros2 launch wildfire_gazebo wildfire_sim.launch.py
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `num_firefighters` | `4` | Number of firefighter ROS namespaces / uAgents |
| `use_vlm` | `true` | Forwarded as `USE_VLM` (scout configuration) |
| `headless` | `true` on macOS, `false` otherwise | Declared for tooling; Gazebo is launched with `-s -r` |
| `image_path` | `WILDFIRE_IMAGE_PATH` or `./wildfire.png` | Aerial image for terrain + initial fire |

Examples:

```bash
ros2 launch wildfire_gazebo wildfire_sim.launch.py num_firefighters:=6
ros2 launch wildfire_gazebo wildfire_sim.launch.py use_vlm:=false image_path:=$(pwd)/wildfire.png
```

If Gemini is unavailable at startup, `fire_grid_node` falls back to a **circular** initial fire blob.

### Foxglove Studio

1. Open Foxglove → **Open connection** → **Foxglove WebSocket** → `ws://localhost:8765` (2D + status).
2. Add another connection → `ws://localhost:8766` (3D scene — same server as `/scene`).
3. Add a **3D** panel, **Fixed frame** `world`, subscribe to `/scene` on the **8766** connection.
4. **Scout fast / slow reasoning (8766, same pattern as `/scene`):** add two **[Log](https://docs.foxglove.dev/docs/visualization/panels/log)** panels. Point one at topic **`/scout_fast_reasoning`** and the other at **`/scout_slow_reasoning`** on the **8766** connection. Both channels use the **`foxglove.Log`** schema (typed MCAP), like `/scene` uses `SceneUpdate`. Fast loop shows live narrative plus `tick_id` / `streaming` in the log line prefix; slow loop shows one pretty-printed JSON snapshot per tactical cycle. **Robot commands always follow the slow loop**, not the fast commentary alone.

ROS topic `/viz/fire_image` remains available for nodes and for debugging; the bridge still receives the rendered bird’s-eye image on `/viz/fire_image`.

## Repository layout

```
BeachHacks2026/
├── setup.sh / run_sim.sh
├── requirements.txt
├── wildfire.png                 # default aerial for world init
└── src/
    ├── wildfire_msgs/           # FireGrid, commands, status
    ├── wildfire_gazebo/         # world, launch, fire_params.yaml
    ├── scout_robot/             # URDF, position controller
    ├── firefighter_robot/       # navigation, water manager
    └── wildfire_agents/         # fire grid, ADK scout, uAgents, ros_bridge,
                                 # foxglove_viz, scene_publisher_3d, sim odometry
```

## ROS packages

| Package | Type | Description |
|---------|------|-------------|
| `wildfire_msgs` | CMake | `FireGrid`, firefighter status, commands |
| `wildfire_gazebo` | CMake | World, `wildfire_sim.launch.py`, `fire_params.yaml` |
| `scout_robot` | Python | Scout URDF + `position_controller` |
| `firefighter_robot` | Python | URDF + `navigation_controller` + `water_manager` |
| `wildfire_agents` | Python | Grid sim, world init (depth + Gemini), ADK scout, uAgents, `ros_bridge`, Foxglove nodes |

## How it works

1. **`fire_grid_node`** simulates fire on a **50×50** grid (see `fire_params.yaml`). At startup, **`world_init`** uses **Gemini** on the aerial image to infer a **12×12** fire mask (mapped into the grid) and **Depth Anything V2** for terrain height; on failure, a circular seed is used.

2. **`foxglove_viz`** renders the bird’s-eye view and streams it on WebSocket **8765**; it also publishes `sensor_msgs/Image` on `/viz/fire_image`.

3. **`scene_publisher_3d`** streams procedural 3D terrain, trees, rocks, fire, and robots on WebSocket **8766** (`/scene`, `/tf`).

4. **`ScoutADKAgent`** runs two Gemini-related cadences: a **fast loop** (`run_reasoning_stream`) streams plain-text situation commentary into a thread-safe buffer and a **timestamped timeline**; a **slow loop** (`run_analysis`) repositions the scout, then either uses a **deterministic** path or **one structured Gemini call** for JSON assignments. The tactical prompt sees only commentary from roughly the **last 10 seconds** of that timeline (override with `SLOW_LOOP_REASONING_WINDOW_SEC`) as **non-authoritative** context (grid telemetry wins on conflict). **`ScoutUAgent`** (port **8000**) carries uAgent messaging; firefighters listen on **8001+**.

5. **`ros_bridge`** runs uAgents on background threads, maps ROS topics ↔ agent commands, publishes JSON on **`/scout/reasoning_status`** and **`/scout/decision_snapshot`** for `foxglove_viz`.

6. **`sim_odom_node`** integrates motion for the demo (no full Gazebo robot plugins required for the grid-driven behavior).

## Tuning fire & agents

**Fire grid** — `src/wildfire_gazebo/config/fire_params.yaml`:

```yaml
fire_grid_node:
  ros__parameters:
    grid_size: 50
    cell_size: 1.0
    base_spread_prob: 0.05
    wind_factor: 2.0
    wind_direction: 45.0    # degrees
    update_rate: 1.0
    initial_fire_radius: 2
    fire_start_x: 0.0
    fire_start_y: 12.0
```

**Bridge / scout intervals** — `src/wildfire_agents/config/agent_config.yaml`: `analysis_interval` (slow loop), `reasoning_interval` (fast loop), `scout_port`, etc. Environment variables `ANALYSIS_INTERVAL` and `REASONING_INTERVAL` override these after install.

## License

Apache-2.0 (see package `license` tags under `src/*/package.xml`).
