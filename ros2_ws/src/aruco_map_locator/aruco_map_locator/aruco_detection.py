"""Functions for detecting ArUco tags and estimating robot poses.

@file        aruco_detection.py
@author      Mowibox (Ousmane THIONGANE)
@version     1.0
@date        2024-12-28
"""

# Imports
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np
import numpy.typing as npt

from .params import *


def detect_aruco(
    img: np.ndarray, camera_matrix: npt.NDArray[np.float64], dist_coeffs: npt.NDArray[np.float64], display: bool = True
) -> tuple:
    """
    Detect the ArUco tags in the provided image.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param display: Displays the markers info if True
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)

    img = img.copy()

    if ids is not None and display:
        marked_img = aruco.drawDetectedMarkers(img, corners, ids, borderColor=(255, 0, 255))
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs)
            marked_img = cv2.drawFrameAxes(marked_img, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE, thickness=2)
        return marked_img, corners, ids
    else:
        return img, corners, ids


def compute_homography(
    img: np.ndarray, camera_matrix: npt.NDArray[np.float64], dist_coeffs: npt.NDArray[np.float64], marker_positions: dict
) -> Optional[npt.NDArray[np.float64]]:
    """
    Compute the homography matrix using the ArUco tags.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    """
    pixel_points: list[npt.NDArray[np.float32]] = []
    real_points: list[npt.NDArray[np.float32]] = []

    img, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions:
            center = corners[i][0].mean(axis=0)
            pixel_points.append(center)
            real_points.append(np.multiply(marker_positions[marker_id], PX_RES))

    if len(pixel_points) < 4:
        return None

    pixel_points_arr = np.asarray(pixel_points, dtype=np.float32)
    real_points_arr = np.asarray(real_points, dtype=np.float32)

    hmtx, _ = cv2.findHomography(pixel_points_arr, real_points_arr)
    return hmtx


def reproject_marker_pos_to_ground(
    tvec: np.ndarray,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    hmtx: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """
    Reprojects the robot ArUco tag position to the ground plane using homography.

    @param marker_pos_cam: The marker position in camera frame
    @param hmtx: The homography matrix
    """
    marker_pos_cam = tvec[0][0]  # Marker position in camera frame

    # Extract rotation from homography
    hmtx_norm = hmtx / hmtx[2, 2]  # Normalize
    h1, h2 = hmtx_norm[:, 0], hmtx_norm[:, 1]

    r1, r2 = h1, h2
    r3 = np.cross(r1, r2)

    # Normalization
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = r3 / np.linalg.norm(r3)  # z-axis direction

    # Delta between marker and robot's top in camera frame
    robot_top_cam = marker_pos_cam - (ROBOT_HEIGHT * r3)

    robot_proj_pos, _ = cv2.projectPoints(robot_top_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)

    robot_ground_pos = np.array([[[robot_proj_pos[0][0][0], robot_proj_pos[0][0][1]]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(robot_ground_pos, hmtx).squeeze()

    return x, y


def compute_camera_pose_from_anchors(
    img: np.ndarray, camera_matrix: npt.NDArray[np.float64], dist_coeffs: npt.NDArray[np.float64], marker_positions: dict
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Compute the camera pose from the anchor ArUco tags.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    """
    _, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    if ids is None:
        return None

    pixel_points = []
    real_points = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions:
            center = corners[i][0].mean(axis=0)
            pixel_points.append(center)
            x, y = marker_positions[marker_id]
            real_points.append([x / M_TO_CM, y / M_TO_CM, 0])

    if len(pixel_points) < 4:
        return None

    pixel_points_arr = np.array(pixel_points, dtype=np.float32)
    real_points_arr = np.array(real_points, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(real_points_arr, pixel_points_arr, camera_matrix, dist_coeffs)

    if not success:
        return None

    return rvec, tvec


def pos_cam_to_world(tvec: np.ndarray, camera_pose: tuple) -> Optional[tuple[float, float]]:
    """
    Transform a position from camera frame to world frame.

    @param tvec: The position in camera frame
    @param camera_pose: The camera pose (rvec, tvec) in world frame
    """
    if camera_pose is None:
        return None

    rvec_cam, tvec_cam = camera_pose
    rmtx_cam, _ = cv2.Rodrigues(rvec_cam)

    marker_pos_cam = tvec[0][0]

    world_z_in_cam = rmtx_cam[:, 2]

    robot_base_cam = marker_pos_cam - (ROBOT_HEIGHT * world_z_in_cam)

    robot_base_world = rmtx_cam.T @ (robot_base_cam - tvec_cam.flatten())

    x, y = robot_base_world[0] * M_TO_CM * PX_RES, robot_base_world[1] * M_TO_CM * PX_RES

    return x, y


def compute_anchor_error(
    img: np.ndarray,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    marker_positions: dict,
    camera_pose: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """
    Compute the error between the detected anchor positions and their real-world positions.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    @param camera_pose: The camera pose (rvec, tvec) in world frame
    """
    if camera_pose is None:
        return 0.0, 0.0

    _, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    if ids is None:
        return 0.0, 0.0

    errors_x = []
    errors_y = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions:
            _, tvec, _ = aruco.estimatePoseSingleMarkers([corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs)

            pos_mes = pos_cam_to_world(tvec, camera_pose)
            if pos_mes is None:
                continue
            else:
                x_mes, y_mes = pos_mes

            x_real, y_real = marker_positions[marker_id]
            x_real *= PX_RES
            y_real *= PX_RES

            errors_x.append(x_mes - x_real)
            errors_y.append(y_mes - y_real)

    if len(errors_x) == 0 or len(errors_y) == 0:
        return 0.0, 0.0

    return np.mean(errors_x), np.mean(errors_y)


def estimate_robot_pose(
    img: np.ndarray,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    marker_positions: dict,
    hmtx: npt.NDArray[np.float64],
) -> dict[int, tuple[float, float, float]]:
    """
    Estimate the robot pose based on ArUco tags.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    @param hmtx: The homography matrix
    """
    img, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    robot_pose: dict[int, tuple[float, float, float]] = {}

    if ids is None:
        return robot_pose

    camera_pose = compute_camera_pose_from_anchors(img, camera_matrix, dist_coeffs, marker_positions)
    x_err, y_err = compute_anchor_error(img, camera_matrix, dist_coeffs, marker_positions, camera_pose)

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions or hmtx is None:  # Don't process the tags used for mapping
            continue

        marker_corners = corners[i]
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers([marker_corners], MARKER_SIZE, camera_matrix, dist_coeffs)

        # Getting position with homography method
        xh, yh = reproject_marker_pos_to_ground(tvec, camera_matrix, dist_coeffs, hmtx)

        # Getting position with PnP
        x, y = xh, yh
        if camera_pose is not None:
            pos = pos_cam_to_world(tvec, camera_pose)
            if pos is not None:
                xp, yp = pos

                x_corr = xp + x_err
                y_corr = yp + y_err

                x = WEIGHT_HMTX * xh + WEIGHT_PNP * x_corr
                y = WEIGHT_HMTX * yh + WEIGHT_PNP * y_corr

        # Getting orientation with angle-axis method
        rot_mtx, _ = cv2.Rodrigues(rvec[0])
        theta_z = np.arctan2(rot_mtx[1, 0], rot_mtx[0, 0])

        print(
            f"Robot n°{marker_id} pose: "
            f"x = {x/PX_RES:.2f} cm; "
            f"y = {y/PX_RES:.2f} cm; "
            f"theta = {np.rad2deg(theta_z):.1f}°"
        )
        robot_pose[marker_id] = (x, y, theta_z)

    return robot_pose


def pose_to_img(robot_pose: dict) -> np.ndarray:
    """
    Display the robot pose into an image.

    @robot_pose: The dictionnary of robot poses
    """
    world_dim = np.multiply(WORLD_SIZE, PX_RES)
    matrix = np.zeros((world_dim[1], world_dim[0], 3), dtype=np.uint8)  # RGB format

    if len(robot_pose) == 0:
        return matrix

    for marker_id, (x, y, theta_z) in robot_pose.items():
        px, py = int(x), int(y)
        py = world_dim[1] - py  # Putting the origin axis to the bottom left

        if 1 <= marker_id <= 5:  # Blue team
            color = (255, 50, 0)
        elif 6 <= marker_id <= 10:  # Yellow team
            color = (0, 205, 255)
        else:
            continue
        neg_color = (255 - color[0], 255 - color[1], 255 - color[2])

        # Robot position
        cv2.circle(matrix, (px, py), int(ROBOT_RADIUS * M_TO_CM * PX_RES), color, -1)

        # Robot orientation
        theta_z = theta_z.item()
        endx = int(px + PX_RES * ROBOT_RADIUS * M_TO_CM * np.cos(theta_z))
        endy = int(py + PX_RES * ROBOT_RADIUS * M_TO_CM * np.sin(theta_z))
        cv2.line(matrix, (px, py), (endx, endy), neg_color, 10)

    return matrix
