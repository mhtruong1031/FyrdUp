#!/bin/bash
# Setup script for Wildfire Simulation
# Works on both Linux (apt + ROS) and macOS (conda/RoboStack + Homebrew).

set -e

echo "========================================="
echo "Wildfire Simulation Setup"
echo "========================================="

OS="$(uname -s)"

# --- ROS 2 -------------------------------------------------------------------

if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS 2 not sourced."
    if [ "$OS" = "Darwin" ]; then
        echo "  Run: eval \"\$(conda shell.zsh hook)\" && conda activate ros2"
    else
        echo "  Run: source /opt/ros/humble/setup.bash"
    fi
    exit 1
fi
echo "✓ ROS 2 $ROS_DISTRO detected"

# --- Gazebo -------------------------------------------------------------------

if ! command -v gz &> /dev/null; then
    echo "Warning: Gazebo (gz) not found in PATH"
    if [ "$OS" = "Darwin" ]; then
        echo "  brew tap osrf/simulation && brew install gz-harmonic"
        echo "  (run with conda deactivated)"
    else
        echo "  sudo apt install gz-harmonic"
    fi
else
    GZ_VERSION=$(gz sim --version 2>&1 | head -1)
    echo "✓ Gazebo found: $GZ_VERSION"
fi

# --- Python -------------------------------------------------------------------

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✓ Python $PYTHON_VERSION detected"

echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# --- Google API key (optional) ------------------------------------------------

if [ -z "$GOOGLE_API_KEY" ]; then
    echo ""
    echo "Note: GOOGLE_API_KEY not set — VLM features disabled."
    echo "  export GOOGLE_API_KEY='your-key-here'"
    echo "  Or run with use_vlm:=false (no key needed)."
else
    echo "✓ GOOGLE_API_KEY is set"
fi

# --- Build --------------------------------------------------------------------

echo ""
echo "Building ROS 2 workspace..."

# --symlink-install fails with conda setuptools >= 69; omit it on macOS.
# Explicitly point CMake at the conda Python so rosidl_generator_py can find it.
if [ "$OS" = "Darwin" ]; then
    colcon build --cmake-args \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DPython_EXECUTABLE="$(which python3)"
else
    colcon build --symlink-install
fi

# On macOS without --symlink-install, executables land in bin/ instead of
# lib/<pkg>/. Symlink them so ros2 launch can find them.
if [ "$OS" = "Darwin" ]; then
    echo "Creating lib/<pkg> symlinks for Python packages..."
    for pkg in scout_robot firefighter_robot wildfire_agents; do
        if [ -d "install/$pkg/bin" ]; then
            mkdir -p "install/$pkg/lib/$pkg"
            for script in install/$pkg/bin/*; do
                [ -f "$script" ] && ln -sf "$(pwd)/$script" "install/$pkg/lib/$pkg/$(basename "$script")"
            done
        fi
    done
fi

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "To run the simulation:"
echo "  source install/setup.bash"
echo "  ros2 launch wildfire_gazebo wildfire_sim.launch.py num_firefighters:=4 use_vlm:=false"
echo ""
