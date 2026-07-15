from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'safe_shared_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='veeti',
    maintainer_email='veeti.pekonen@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator_node = safe_shared_control.simulator_node:main',
            'controller_node = safe_shared_control.controller_node:main',
            'keyboard_teleop_node = safe_shared_control.keyboard_teleop_node:main',
            'mpc_controller_node = safe_shared_control.afs_mpc_controller_node:main',
        ],
    },
)
