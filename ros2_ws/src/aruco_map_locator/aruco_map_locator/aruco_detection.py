"""
    @file        aruco_detection.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Code for detcting the aruco tags
    @version     1.0
    @date        2024-12-28
    
"""

# Imports
import cv2
import yaml
import numpy as np
from params import *
import cv2.aruco as aruco
import matplotlib.pyplot as plt 

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


def load_img(path: str, size: tuple[int, int]=None) -> np.ndarray:
    """
    Loads an image and resizes it if specified

    @param path: The image filepath
    @param size: The size of the resized image (w, h)
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size) 
    return img


def display_img(img: np.ndarray, img2: np.ndarray = None, grid: bool = False) -> None:
    """
    Displays one or two images on a plot, optionally with a grid overlay

    @param img: The first input image
    @param img2: The second image (optional)
    @param grid: Displays a grid if True
    """
    n_cols = 2 if img2 is not None else 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    def plot_image(ax, img, grid):
        height, width = img.shape[:2]
        if grid:
            for x in range(width + 1):
                ax.axvline(x - 0.5, linewidth=0.5)
            for y in range(height + 1):
                ax.axhline(y - 0.5, linewidth=0.5)
        ax.imshow(img)

    plot_image(axes[0], img, grid)

    if img2 is not None:
        plot_image(axes[1], img2, grid)

    plt.tight_layout()
    plt.show()
    

def detect_aruco(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, display: bool=True) -> tuple: 
    """
    Detects the aruco tags in the provided image

    @param img: The input image 
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param display: Displays the markers info if True
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(
       gray, aruco_dict, parameters=arucoParameters)
    
    img = img.copy()
    
    if np.all(ids) and display:
        marked_img = aruco.drawDetectedMarkers(img, corners, ids, borderColor=(255 ,0, 255))
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, camera_matrix, dist_coeffs)
            marked_img = cv2.drawFrameAxes(marked_img, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE, thickness=2)
        return marked_img, corners, ids
    else:
        return img, corners, ids
    


def compute_homography(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, marker_positions: dict) -> tuple:
    """
    Computes the homography matrix using the aruco tags

    @param img: The input image 
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    """
    pixel_points = []
    real_points = []

    img, corners, ids =  detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions:
            center = corners[i][0].mean(axis=0)
            pixel_points.append(center)
            real_points.append(np.multiply(marker_positions[marker_id], PX_RES))

    if len(pixel_points) < 4:
        return img, None
    
    pixel_points = np.array(pixel_points)
    real_points = np.array(real_points)

    hmtx, _ = cv2.findHomography(pixel_points, real_points)
    output_size = np.multiply(WORLD_SIZE, PX_RES)
    warped_image = cv2.warpPerspective(img, hmtx, output_size)
    return warped_image, hmtx


def estimate_robot_pose(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, marker_positions: dict, hmtx: np.ndarray) -> tuple:
    """
    Estimates the robot pose based on aruco tag

    @param img: The input image 
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    @param hmtx: The homography matrix
    """
    img, corners, ids =  detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    robot_pose = {}

    if np.all(ids):
        return robot_pose
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions: # Don't process the tags used for mapping
            continue
        

        marker_corners = corners[i]
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, camera_matrix, dist_coeffs)

        # Getting position
        marker_corners_in_h = cv2.perspectiveTransform(np.array(marker_corners).reshape(-1, 1, 2), hmtx).squeeze()
        x, y = marker_corners_in_h.mean(axis=0) # Coordinates of the center

        # Getting orientation with angle-axis method
        rot_mtx, _ = cv2.Rodrigues(rvec[0])
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mtx, tvec[0].reshape(-1, 1))))[6]
        theta_z = euler_angles[2]

        robot_pose[marker_id] = (x, y, theta_z)

    return robot_pose


def pose_to_img(robots_pose: dict) -> np.ndarray:
    """
    Displays the robot pose into an image

    @robots_pose: The dictionnary of robot poses
    """
    world_dim = np.multiply(WORLD_SIZE, PX_RES)
    matrix = np.zeros((world_dim[1], world_dim[0], 3), dtype=np.uint8) # RGB format

    for marker_id, (x, y, theta_z) in robots_pose.items():
        px, py = int(x), int(y)

        if 1 <= marker_id <= 5:
            color = (0, 50, 255)
        elif 6 <= marker_id <= 6:
            color = (255, 205, 0)
        else:
            continue
        neg_color = (255-color[0], 255-color[1], 255-color[2])

        # Robot position
        cv2.circle(matrix, (px, py), ROBOT_RADIUS*PX_RES, color, -1)

        # Robot orientation
        theta_z = theta_z.item()
        endx = int(px + PX_RES*ROBOT_RADIUS*np.cos(theta_z))
        endy = int(py + PX_RES*ROBOT_RADIUS*np.sin(theta_z))
        cv2.line(matrix, (px, py), (endx, endy), neg_color, 10)

    return matrix