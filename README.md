# Wildfire Simulation — Gazebo + Fetch uAgents + Gemini VLM

A multi-agent wildfire fighting simulation built on ROS 2 Humble, Gazebo Harmonic,
and the Fetch uAgent SDK. A scout drone uses Google Gemini (VLM) to analyse the
fire from above and coordinates N firefighter ground robots in real time.

## Architecture

```
┌─────────────────────────────────────────────┐
│          Gazebo Harmonic (headless)          │
│   20×20 m ground plane + water supply        │
└─────────────────────────────────────────────┘
                    │
             ROS 2 Humble topics
                    │
    ┌───────────────┼───────────────┐
    │               │               │
fire_grid_node  scout_robot   firefighter_robot (×N)
  /fire_grid    /scout/*      /firefighter_*/cmd_vel
  /water_spray                /firefighter_*/water_level
                    │
              ros_bridge
           (ROS ↔ uAgent)
                    │
         ┌──────────┴──────────┐
      scout uAgent       firefighter uAgents
      (Gemini VLM)       (move / spray / refill)
```

## Prerequisites

| Component | Version |
|-----------|---------|
| ROS 2 | Humble |
| Gazebo | Harmonic (gz-harmonic) |
| Python | 3.10+ |
| Google API key | For Gemini VLM |

### macOS (Apple Silicon)

```bash
# Deactivate conda first if active
conda deactivate

# Install Gazebo Harmonic via Homebrew
brew tap osrf/simulation
brew install gz-harmonic

# Verify
gz sim --version   # expect 8.x.x
```

### RoboStack conda environment (recommended)

```bash
conda create -n wildfire python=3.11
conda activate wildfire
conda install -c robostack-staging ros-humble-desktop
conda install -c robostack-staging ros-humble-ros-gz
```

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

## Packages

| Package | Type | Description |
|---------|------|-------------|
| `wildfire_msgs` | CMake | Custom ROS 2 messages: FireGrid, FirefighterStatus, MoveCommand, RefillCommand |
| `wildfire_gazebo` | CMake | Gazebo world, launch file, fire parameters |
| `scout_robot` | Python | Scout drone URDF + position controller |
| `firefighter_robot` | Python | Firefighter URDF + navigation controller + water manager |
| `wildfire_agents` | Python | uAgent logic, Gemini VLM, fire grid sim, viz renderer, ROS bridge |

## How It Works

1. **Fire Grid Node** simulates fire spread on a 20×20 grid with wind-biased
   probabilistic spreading. Fire starts as a blob at grid center.

2. **Viz Renderer** subscribes to the fire grid and renders a bird's-eye image
   showing fire (red), unburned terrain (green), firefighters (yellow), and the
   water supply (blue). Published on `/viz/fire_image`.

3. **Scout Agent** (uAgent on port 8000) periodically captures the bird's-eye
   image and sends it to **Gemini 2.0 Flash** for tactical analysis. The VLM
   returns fire locations, recommended positions, and threat level as JSON.

4. **Scout allocates firefighters** using greedy nearest-neighbour matching
   between available firefighters and recommended positions.

5. **Firefighter Agents** (uAgents on ports 8001+) receive `MoveCommand` and
   `RefillCommand` messages. They navigate to targets, spray water at fires,
   and return to the water supply when low on water.

6. **ROS Bridge** runs all uAgents in daemon threads and translates between
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
