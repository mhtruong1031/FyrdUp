# Wildfire Simulation — Gazebo + Fetch uAgents + Gemini VLM

A multi-agent wildfire fighting simulation built on ROS 2 Humble, Gazebo Harmonic,
and the Fetch uAgent SDK. A scout drone uses Google Gemini (VLM) to analyse the
fire from above and coordinates N firefighter ground robots in real time.

Gazebo Harmonic runs headlessly (server-only, no GUI) with visualization handled
by **Foxglove Studio** over a WebSocket bridge. A dedicated 3D scene publisher
generates procedural terrain, trees, rocks, fire volumes, and robot markers so
the full simulation is visible in Foxglove's 3D panel.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Gazebo Harmonic (headless, local)             │
│   20×20 m procedural terrain + boulders + water supply  │
└─────────────────────────────────────────────────────────┘
                          │
                   ROS 2 Humble topics
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
fire_grid_node      scout_robot       firefighter_robot (×N)
  /fire_grid        /scout/*          /firefighter_*/cmd_vel
  /water_spray                        /firefighter_*/water_level
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                scene_publisher_3d
          /viz/terrain_markers   (MarkerArray)
          /viz/tree_markers      (MarkerArray)
          /viz/rock_markers      (MarkerArray)
          /viz/fire_markers      (MarkerArray)
          /viz/robot_markers     (MarkerArray)
          /tf                    (static world→map)
                          │
                   foxglove_bridge
                  ws://localhost:8765
                          │
                   Foxglove Studio
                          │
                    ros_bridge
                 (ROS ↔ uAgent)
                          │
           ┌──────────────┴──────────────┐
        scout uAgent           firefighter uAgents
        (Gemini VLM)           (move / spray / refill)
```

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| ROS 2 | Humble | via RoboStack or native install |
| Gazebo | Harmonic (gz-harmonic) | installed locally via Homebrew or apt |
| Python | 3.10+ | |
| Foxglove Studio | Latest | Desktop app or web — connects to `ws://localhost:8765` |
| foxglove_bridge | ROS 2 package | `apt install ros-humble-foxglove-bridge` or conda equivalent |
| Google API key | — | For Gemini VLM |

### Gazebo Harmonic (local install)

Gazebo Harmonic must be installed directly on the host machine (not inside
Docker or conda). It runs in server-only mode (`gz sim -s -r`) — no GPU or
display server is required.

**macOS (Apple Silicon):**

```bash
# Deactivate conda first — conda and Homebrew system libs conflict
conda deactivate

brew tap osrf/simulation
brew install gz-harmonic

# Verify
gz sim --version   # expect 8.x.x
```

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install gz-harmonic
gz sim --version
```

### RoboStack conda environment (recommended for ROS 2)

```bash
conda create -n wildfire python=3.11
conda activate wildfire
conda install -c robostack-staging ros-humble-desktop
conda install -c robostack-staging ros-humble-ros-gz
conda install -c robostack-staging ros-humble-foxglove-bridge
```

### Foxglove Studio

Download the desktop app from <https://foxglove.dev/download> or use the web
version at <https://app.foxglove.dev>. No install is needed on the simulation
host — Foxglove connects over WebSocket from any machine on the same network.

## Setup

```bash
cd /path/to/BeachHacks2026

# Install Python dependencies
pip install -r requirements.txt

# Build the ROS 2 workspace
colcon build --cmake-args -DPython3_EXECUTABLE=$(which python3)

# Fix executable paths (conda/macOS quirk)
for pkg in scout_robot firefighter_robot wildfire_agents; do
    mkdir -p "install/$pkg/lib/$pkg"
    for script in install/$pkg/bin/*; do
        [ -f "$script" ] && ln -sf "$(pwd)/$script" "install/$pkg/lib/$pkg/$(basename "$script")"
    done
done

# Source the workspace
source install/setup.bash
```

## Running

```bash
# Set your Google API key (required for VLM mode)
export GOOGLE_API_KEY="your-api-key-here"

# Launch with defaults (4 firefighters, VLM enabled)
ros2 launch wildfire_gazebo wildfire_sim.launch.py

# Custom number of firefighters
ros2 launch wildfire_gazebo wildfire_sim.launch.py num_firefighters:=6

# Without VLM (uses grid-based fallback analysis)
ros2 launch wildfire_gazebo wildfire_sim.launch.py use_vlm:=false
```

The launch file automatically starts Gazebo in headless mode on macOS. It also
launches `foxglove_bridge` (port 8765) and `scene_publisher_3d` for 3D
visualization.

### Viewing in Foxglove Studio

1. Open Foxglove Studio
2. Choose **Open connection** > **Foxglove WebSocket** > `ws://localhost:8765`
3. Add a **3D** panel
4. Set **Fixed frame** to `world`
5. The panel auto-discovers marker topics — you should see procedural terrain,
   trees, rocks, fire spreading, and robot positions updating in real time

Key topics for the 3D panel:

| Topic | Type | Content |
|-------|------|---------|
| `/viz/terrain_markers` | MarkerArray | 20×20 noise-based terrain tiles |
| `/viz/tree_markers` | MarkerArray | ~100 procedurally placed trees |
| `/viz/rock_markers` | MarkerArray | ~30 scattered rock formations |
| `/viz/fire_markers` | MarkerArray | Volumetric fire cubes (height = intensity) |
| `/viz/robot_markers` | MarkerArray | Firefighters, scout drone, water supply |
| `/viz/fire_image` | Image | 2D bird's-eye view (used by VLM) |

## Packages

| Package | Type | Description |
|---------|------|-------------|
| `wildfire_msgs` | CMake | Custom ROS 2 messages: FireGrid, FirefighterStatus, MoveCommand, RefillCommand |
| `wildfire_gazebo` | CMake | Gazebo world (with procedural rocks/mounds), launch file, fire parameters |
| `scout_robot` | Python | Scout drone URDF + position controller |
| `firefighter_robot` | Python | Firefighter URDF + navigation controller + water manager |
| `wildfire_agents` | Python | uAgent logic, Gemini VLM, fire grid sim, viz renderer, 3D scene publisher, ROS bridge |

## How It Works

1. **Fire Grid Node** simulates fire spread on a 20×20 grid with wind-biased
   probabilistic spreading. Fire starts as a blob at grid center.

2. **Viz Renderer** subscribes to the fire grid and renders a bird's-eye image
   showing fire (red), unburned terrain (green), firefighters (yellow), and the
   water supply (blue). Published on `/viz/fire_image`.

3. **3D Scene Publisher** generates a procedural 3D environment (terrain with
   fractal noise height, ~100 trees via rejection sampling, ~30 rocks) and
   converts the fire grid + robot odometry into `MarkerArray` messages that
   Foxglove Studio renders in its 3D panel.

4. **Scout Agent** (uAgent on port 8000) periodically captures the bird's-eye
   image and sends it to **Gemini 2.0 Flash** for tactical analysis. The VLM
   returns fire locations, recommended positions, and threat level as JSON.

5. **Scout allocates firefighters** using greedy nearest-neighbour matching
   between available firefighters and recommended positions.

6. **Firefighter Agents** (uAgents on ports 8001+) receive `MoveCommand` and
   `RefillCommand` messages. They navigate to targets, spray water at fires,
   and return to the water supply when low on water.

7. **ROS Bridge** runs all uAgents in daemon threads and translates between
   ROS topics and uAgent messages.

## Configuration

Edit `src/wildfire_gazebo/config/fire_params.yaml`:

```yaml
fire_grid_node:
  ros__parameters:
    grid_size: 20
    cell_size: 1.0
    base_spread_prob: 0.05
    wind_factor: 2.0
    wind_direction: 45.0    # degrees (0=N, 90=E)
    update_rate: 1.0
    initial_fire_radius: 2
```

## License

Apache-2.0
