"""Parameters file for the aruco_map_locator package.

@file        params.py
@author      Mowibox (Ousmane THIONGANE)
@version     1.0
@date        2024-12-29
"""

# Imports
import os

import numpy as np
import numpy.typing as npt
import yaml
from ament_index_python.packages import get_package_share_directory  # type: ignore

package_share_dir = get_package_share_directory("image_provider")
CALIBRATION_FILE = os.path.join(package_share_dir, "config", "cam_params.yaml")

M_TO_CM = 100  # Meters to centimeters conversion

ROBOT_RADIUS = 0.01  # The robot radius for vizualization (m)
ROBOT_HEIGHT = 0.035  # The robot height (m)

MARKER_SIZE = 0.02  # In meters (2 cm)
WORLD_SIZE = [0.30, 0.20]  # World dimensions (w, h), (m)

# Aruco real-world coordinates ID:[x, y] (m)
MARKER_POSITIONS = {
    20: [0.06, 0.14],
    21: [0.24, 0.14],
    22: [0.06, 0.06],
    23: [0.24, 0.06],
}

WEIGHT_HMTX = 0.5  # Weight for the homography matrix position estimation
WEIGHT_PNP = 0.5  # Weight for the PnP posiiton estimation

# For visualization purposes
IMAGE_WIDTH = 600  # Image width (px)
IMAGE_HEIGHT = 400  # Image height (px)


def load_calibration_params(filepath: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Loads the camera calibration parameters specified in the provided yaml file.

    @param filepath: The yaml file path
    """
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"][0])

    return camera_matrix, dist_coeffs


camera_matrix, dist_coeffs = load_calibration_params(CALIBRATION_FILE)
