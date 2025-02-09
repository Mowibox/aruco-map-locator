"""
    @file        params.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Parameters file
    @version     1.0
    @date        2024-12-29
    
"""
# Imports
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory

package_share_dir = get_package_share_directory('image_provider')
CALIBRATION_FILE = os.path.join(package_share_dir, 'config', 'cam_params.yaml')

ROBOT_RADIUS = 0.01      # The robot radius (in m)
ROBOT_HEIGHT = 0.035     # The robot height (in m)

MARKER_SIZE = 0.02       # In meters (2 cm)
M_TO_CM = 100            # Meters to centimeters conversion
WORLD_SIZE = [30, 20]    # World dimensions (w, h) 
PX_RES = 20              # 1 cm is 1*PX_RES pixels

# Aruco real-world coordinates ID:[x, y] (in cm)
MARKER_POSITIONS = {
    20: [6, 14],
    21: [24, 14],
    22: [6, 6],
    23: [24, 6],
}

def load_calibration_params(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the camera calibration parameters specified in the provided yaml file
    @param filepath: The yaml file path 
    """
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['distortion_coefficients'][0])

    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_calibration_params(CALIBRATION_FILE)
