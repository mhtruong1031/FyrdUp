from setuptools import setup
import os
from glob import glob

package_name = 'firefighter_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wildfire_sim',
    maintainer_email='user@example.com',
    description='Firefighter ground robot for wildfire simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_controller = firefighter_robot.navigation_controller:main',
            'water_manager = firefighter_robot.water_manager:main',
        ],
    },
)
