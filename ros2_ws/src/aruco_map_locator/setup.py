from setuptools import find_packages, setup

package_name = 'aruco_map_locator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/aruco_map_locator_launch.py']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pyyaml',
        'opencv-python',
        'opencv-contrib-python'
        ],
    zip_safe=True,
    maintainer='mowibox',
    maintainer_email='ousmane.thiongane@ensea.fr',
    description='ArUco image processing and map generation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aruco_processing = aruco_map_locator.aruco_processing:main', 
        ],
    },
)
