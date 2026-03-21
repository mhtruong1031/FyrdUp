#!/usr/bin/env bash
set -euo pipefail

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
cd "$SCRIPT_DIR/install" && source local_setup.bash
cd "$SCRIPT_DIR"

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
