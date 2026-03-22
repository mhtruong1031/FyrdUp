#!/usr/bin/env bash
set -eo pipefail

# ── Kill any existing simulation processes ──────────────────────────────────
echo "[run_sim] Stopping existing processes..."
pkill -f "ros2 launch" 2>/dev/null || true
pkill -f "fire_grid_node" 2>/dev/null || true
pkill -f "ros_bridge" 2>/dev/null || true
pkill -f "scene_publisher" 2>/dev/null || true
pkill -f "foxglove_viz" 2>/dev/null || true
pkill -f "navigation_controller" 2>/dev/null || true
pkill -f "water_manager" 2>/dev/null || true
pkill -f "position_controller" 2>/dev/null || true
pkill -f "sim_odom_node" 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
sleep 2

# ── Resolve paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"

# ── Activate conda + ROS 2 overlay ─────────────────────────────────────────
source "$CONDA_SH"
conda activate ros2

# ── Rebuild packages ──────────────────────────────────────────────────────
echo "[run_sim] Building workspace..."
cd "$SCRIPT_DIR"
CONDA_PREFIX="${CONDA_PREFIX:-/opt/homebrew/Caskroom/miniconda/base/envs/ros2}"
if ! colcon build \
    --parallel-workers "$(sysctl -n hw.ncpu)" \
    --cmake-args \
        -DPython_ROOT_DIR="$CONDA_PREFIX" \
        -DPython_FIND_STRATEGY=LOCATION \
        -DPython_LIBRARY="$CONDA_PREFIX/lib/libpython3.12.dylib" \
        -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python3" \
    2>&1 | tail -n 8; then
    echo "[run_sim] ERROR: colcon build failed — aborting."
    exit 1
fi

# Executables land in bin/ without --symlink-install; ROS 2 expects lib/<pkg>/
for pkg in scout_robot firefighter_robot wildfire_agents; do
    mkdir -p "install/$pkg/lib/$pkg"
    for script in install/$pkg/bin/*; do
        [ -f "$script" ] && ln -sf "$SCRIPT_DIR/$script" "install/$pkg/lib/$pkg/$(basename "$script")"
    done
done

cd "$SCRIPT_DIR/install" && source local_setup.bash && cd "$SCRIPT_DIR"

# ── Environment variables ──────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi
export WILDFIRE_IMAGE_PATH="${WILDFIRE_IMAGE_PATH:-$SCRIPT_DIR/wildfire.png}"

# ── Configurable launch args (override via env or CLI) ─────────────────────
NUM_FF="${NUM_FIREFIGHTERS:-4}"
USE_VLM="${USE_VLM:-true}"

echo "[run_sim] WILDFIRE_IMAGE_PATH=$WILDFIRE_IMAGE_PATH"
echo "[run_sim] NUM_FIREFIGHTERS=$NUM_FF  USE_VLM=$USE_VLM"
echo "[run_sim] Launching..."

# ── Launch ─────────────────────────────────────────────────────────────────
ros2 launch wildfire_gazebo wildfire_sim.launch.py \
    num_firefighters:="$NUM_FF" \
    use_vlm:="$USE_VLM"
