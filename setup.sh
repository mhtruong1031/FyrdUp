#!/usr/bin/env bash
# One-shot setup: system deps (Ubuntu 22.04), Python packages, colcon build.
# macOS: install ROS (e.g. RoboStack conda) + Homebrew gz-harmonic first, then run.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

OS="$(uname -s)"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# --- Ubuntu 22.04: ROS 2 Humble + Gazebo Harmonic + tools --------------------

bootstrap_ubuntu_jammy() {
  echo ""
  echo ">>> Bootstrapping system packages (Ubuntu 22.04 jammy)..."
  export DEBIAN_FRONTEND=noninteractive

  $SUDO apt-get update
  $SUDO apt-get install -y \
    software-properties-common \
    curl \
    wget \
    gnupg \
    lsb-release \
    ca-certificates

  $SUDO add-apt-repository -y universe
  $SUDO apt-get update

  if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    $SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
  fi
  if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      | $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null
  fi

  if [ ! -f /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg ]; then
    $SUDO wget -q https://packages.osrfoundation.org/gazebo.gpg \
      -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  fi
  if [ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      | $SUDO tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
  fi

  $SUDO apt-get update
  $SUDO apt-get install -y \
    ros-humble-desktop \
    ros-humble-ros-gz-sim \
    python3-colcon-common-extensions \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gz-harmonic

  echo ">>> System packages installed."
}

ensure_ros_sourced_linux() {
  if [ -n "${ROS_DISTRO:-}" ]; then
    return 0
  fi
  if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
    return 0
  fi
  if [ ! -f /etc/os-release ]; then
    echo "Error: ROS 2 not found and /etc/os-release missing."
    exit 1
  fi
  # shellcheck source=/dev/null
  . /etc/os-release
  if [ "$ID" = "ubuntu" ] && [ "${VERSION_ID:-}" = "22.04" ]; then
    bootstrap_ubuntu_jammy
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
    return 0
  fi
  echo "Error: ROS 2 Humble not installed or not sourced."
  echo "  Official debs target Ubuntu 22.04 (jammy). Install ROS + Gazebo, then:"
  echo "    source /opt/ros/humble/setup.bash"
  echo "    ./setup.sh"
  echo "  Docs: https://docs.ros.org/en/humble/Installation.html"
  exit 1
}

# --- Main ---------------------------------------------------------------------

echo "========================================="
echo "Wildfire Simulation Setup"
echo "========================================="

if [ "$OS" = "Linux" ]; then
  ensure_ros_sourced_linux
else
  if [ -z "${ROS_DISTRO:-}" ]; then
    echo "Error: ROS 2 not sourced."
    echo "  macOS: eval \"\$(conda shell.zsh hook)\" && conda activate ros2"
    echo "  Then: brew tap osrf/simulation && brew install gz-harmonic  (conda deactivated)"
    exit 1
  fi
fi
echo "✓ ROS 2 ${ROS_DISTRO} detected"

# --- Gazebo -------------------------------------------------------------------

if ! command -v gz &> /dev/null; then
  echo "Error: Gazebo (gz) not found in PATH."
  if [ "$OS" = "Darwin" ]; then
    echo "  conda deactivate && brew tap osrf/simulation && brew install gz-harmonic"
  else
    echo "  Re-run on Ubuntu 22.04 so ./setup.sh can install gz-harmonic, or: sudo apt install gz-harmonic"
  fi
  exit 1
fi
GZ_VERSION=$(gz sim --version 2>&1 | head -1)
echo "✓ Gazebo found: $GZ_VERSION"

# --- Python deps --------------------------------------------------------------
# Linux + PEP 668: venv with --system-site-packages so ROS 2 apt Python (rclpy) stays visible.
# macOS: Homebrew PEP 668 — use conda ROS env (see error below), not an isolated venv.

if [ "$OS" = "Darwin" ]; then
  if python3 -c "import pathlib, sys; sys.exit(0 if (pathlib.Path(sys.prefix) / 'externally-managed').exists() else 1)"; then
    echo ""
    echo "Error: this Python is externally managed (PEP 668) — pip cannot install here."
    echo "  On macOS with Homebrew Python, activate your ROS conda env first, then re-run:"
    echo "    conda activate ros2"
    echo "    ./setup.sh"
    echo "  Or use a venv with access to your ROS Python (not recommended unless you know the layout)."
    exit 1
  fi
elif python3 -c "import pathlib, sys; sys.exit(0 if (pathlib.Path(sys.prefix) / 'externally-managed').exists() else 1)"; then
  if [ ! -d ".venv" ]; then
    echo "Creating .venv --system-site-packages (PEP 668; keeps ROS 2 system packages visible)..."
    python3 -m venv --system-site-packages .venv
  fi
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "✓ Using venv: $VIRTUAL_ENV"
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✓ Python $PYTHON_VERSION ($(command -v python3))"

echo ""
echo "Installing Python dependencies..."
python3 -m pip install -U pip wheel
python3 -m pip install -r requirements.txt

# --- Google API key (optional) -----------------------------------------------

if [ -z "${GOOGLE_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo ""
  echo "Note: GOOGLE_API_KEY / GEMINI_API_KEY not set — VLM features disabled."
  echo "  Add keys to .env or export GOOGLE_API_KEY='your-key-here'"
  echo "  Or run with use_vlm:=false (no key needed)."
else
  echo "✓ Gemini API key is set (GOOGLE_API_KEY or GEMINI_API_KEY)"
fi

# --- Build --------------------------------------------------------------------

echo ""
echo "Building ROS 2 workspace..."

if [ "$OS" = "Darwin" ]; then
  colcon build --cmake-args \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DPython_EXECUTABLE="$(which python3)"
else
  colcon build --symlink-install --cmake-args \
    "-DPython3_EXECUTABLE=$(which python3)" \
    "-DPython_EXECUTABLE=$(which python3)"
fi

if [ "$OS" = "Darwin" ]; then
  echo "Creating lib/<pkg> symlinks for Python packages..."
  for pkg in scout_robot firefighter_robot wildfire_agents; do
    if [ -d "install/$pkg/bin" ]; then
      mkdir -p "install/$pkg/lib/$pkg"
      for script in install/"$pkg"/bin/*; do
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
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "This workspace uses a Python venv. Before ros2 launch, run:"
  echo "  source .venv/bin/activate"
  echo ""
fi
echo "To run the simulation:"
echo "  source install/setup.bash"
echo "  ros2 launch wildfire_gazebo wildfire_sim.launch.py num_firefighters:=4 use_vlm:=false"
echo ""
