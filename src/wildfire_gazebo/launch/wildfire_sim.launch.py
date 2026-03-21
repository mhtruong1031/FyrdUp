#!/usr/bin/env python3
"""
Main launch file for the wildfire simulation.

Launches Gazebo Harmonic (headless on macOS), fire grid node, scout controller,
N firefighter controllers + water managers, bird's-eye viz renderer, and the
ROS-uAgent bridge.
"""

import os
import platform

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_firefighters(context, *_args, **_kwargs):
    """Dynamically spawn N firefighter navigation + water manager nodes."""
    n = int(context.launch_configurations['num_firefighters'])
    actions = []
    for i in range(n):
        ff_name = f'firefighter_{i + 1}'
        actions.append(Node(
            package='firefighter_robot',
            executable='navigation_controller',
            name='navigation_controller',
            namespace=ff_name,
            output='screen',
        ))
        actions.append(Node(
            package='firefighter_robot',
            executable='water_manager',
            name='water_manager',
            namespace=ff_name,
            output='screen',
        ))
    return actions


def generate_launch_description():
    wildfire_gazebo_dir = get_package_share_directory('wildfire_gazebo')
    scout_robot_dir = get_package_share_directory('scout_robot')

    # -- launch arguments ----------------------------------------------------

    num_ff_arg = DeclareLaunchArgument(
        'num_firefighters', default_value='4',
        description='Number of firefighter robots')

    use_vlm_arg = DeclareLaunchArgument(
        'use_vlm', default_value='true',
        description='Use Gemini VLM for fire analysis (requires GOOGLE_API_KEY)')

    headless_default = 'true' if platform.system() == 'Darwin' else 'false'
    headless_arg = DeclareLaunchArgument(
        'headless', default_value=headless_default,
        description='Run Gazebo in server-only mode (no GUI)')

    # -- environment for ros_bridge ------------------------------------------

    set_num_ff = SetEnvironmentVariable(
        'NUM_FIREFIGHTERS', LaunchConfiguration('num_firefighters'))
    set_use_vlm = SetEnvironmentVariable(
        'USE_VLM', LaunchConfiguration('use_vlm'))

    # -- Gazebo Harmonic (server-only on macOS) ------------------------------

    world_path = os.path.join(wildfire_gazebo_dir, 'worlds', 'wildfire.world')
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', world_path],
        output='screen',
    )

    # -- Scout position controller -------------------------------------------

    scout_controller = Node(
        package='scout_robot',
        executable='position_controller',
        name='scout_position_controller',
        namespace='scout',
        parameters=[
            os.path.join(scout_robot_dir, 'config', 'camera_params.yaml'),
        ],
        output='screen',
    )

    # -- Fire grid node (replaces Classic world plugin) ----------------------

    fire_grid = Node(
        package='wildfire_agents',
        executable='fire_grid_node',
        name='fire_grid_node',
        parameters=[
            os.path.join(wildfire_gazebo_dir, 'config', 'fire_params.yaml'),
        ],
        output='screen',
    )

    # -- Foxglove visualization (2D bird's-eye + foxglove-sdk server) --------

    foxglove_viz = Node(
        package='wildfire_agents',
        executable='foxglove_viz',
        name='foxglove_viz',
        output='screen',
    )

    # -- 3D scene publisher (foxglove-sdk SceneUpdate) ----------------------

    scene_publisher_3d = Node(
        package='wildfire_agents',
        executable='scene_publisher_3d',
        name='scene_publisher_3d',
        output='screen',
    )

    # -- ROS-uAgent bridge ---------------------------------------------------

    ros_bridge = Node(
        package='wildfire_agents',
        executable='ros_bridge',
        name='ros_bridge',
        output='screen',
    )

    # -- assemble ------------------------------------------------------------

    ld = LaunchDescription()

    ld.add_action(num_ff_arg)
    ld.add_action(use_vlm_arg)
    ld.add_action(headless_arg)
    ld.add_action(set_num_ff)
    ld.add_action(set_use_vlm)

    ld.add_action(gazebo)
    ld.add_action(scout_controller)
    ld.add_action(fire_grid)
    ld.add_action(OpaqueFunction(function=_spawn_firefighters))
    ld.add_action(foxglove_viz)
    ld.add_action(scene_publisher_3d)
    ld.add_action(ros_bridge)

    return ld
