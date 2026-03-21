from setuptools import setup
import os
from glob import glob

package_name = 'wildfire_agents'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wildfire_sim',
    maintainer_email='user@example.com',
    description='Fetch uAgent integration for wildfire simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scout_agent = wildfire_agents.scout_agent:main',
            'firefighter_agent = wildfire_agents.firefighter_agent:main',
            'ros_bridge = wildfire_agents.ros_bridge:main',
            'fire_grid_node = wildfire_agents.fire_grid_node:main',
            'viz_renderer = wildfire_agents.viz_renderer:main',
            'foxglove_viz = wildfire_agents.foxglove_viz:main',
            'scene_publisher_3d = wildfire_agents.scene_publisher_3d:main',
        ],
    },
)
