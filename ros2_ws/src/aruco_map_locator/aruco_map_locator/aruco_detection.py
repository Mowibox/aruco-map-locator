"""Functions for detecting ArUco tags and estimating robot poses.

@file        aruco_detection.py
@author      Mowibox (Ousmane THIONGANE)
@version     1.0
@date        2024-12-28

Coding convention for frame transformations:
- pos_a_b: position of frame b expressed in frame a
- Rmtx_a_b: rotation matrix from frame b to frame a
- tvec_a_b: translation vector from frame b to frame a
- rvec_a_b: rotation vector from frame b to frame a
"""

# Imports
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np
import numpy.typing as npt

from .params import *


def detect_aruco(
    img: np.ndarray, camera_matrix: npt.NDArray[np.float32], dist_coeffs: npt.NDArray[np.float32], display: bool = True
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
    img: np.ndarray, camera_matrix: npt.NDArray[np.float32], dist_coeffs: npt.NDArray[np.float32], marker_positions: dict
) -> Optional[npt.NDArray[np.float64]]:
    """
    Compute the homography matrix using the ArUco tags.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    """
    image_points: list[npt.NDArray[np.float32]] = []  # [x', y']
    object_points: list[npt.NDArray[np.float32]] = []  # [X, Y]

    img, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    if ids is None:
        return None
    
    half_size = MARKER_SIZE / 2.0
    corners_2d_loc = np.array(
        [
            [-half_size, +half_size],  # Top-left
            [+half_size, +half_size],  # Top-right
            [+half_size, -half_size],  # Bottom-right
            [-half_size, -half_size],  # Bottom-left
        ],
        dtype=np.float32,
    )

    if ids is not None and corners is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_positions:
                marker_corners_2d = corners[i][0] 
                x, y = marker_positions[marker_id]

                marker_corners_world = corners_2d_loc + np.array([x, y], dtype=np.float32)

                for corner_img, corner_world in zip(marker_corners_2d, marker_corners_world):
                    image_points.append(corner_img)
                    object_points.append(corner_world)

    if len(image_points) < 4:
        return None

    image_points_arr = np.asarray(image_points, dtype=np.float32)
    object_points_arr = np.asarray(object_points, dtype=np.float32)

    # lambda * [x', y', 1].T = Hmtx * [X, Y, 1].T
    Hmtx, _ = cv2.findHomography(image_points_arr, object_points_arr)

    if Hmtx is None:
        return None

    return Hmtx.astype(np.float64)


def reproject_marker_pos_to_ground(
    tvec: np.ndarray,
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    Hmtx: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """
    Reprojects the robot ArUco tag position to the ground plane using homography.

    @param tvec: The translation vector of the marker
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param Hmtx: The homography matrix
    """
    pos_cam_marker = tvec[0][0]  # Marker position in camera frame [xc, yc, zc]

    # Extract R_cam_ground from homography
    # H(z=0) = [r1 r2 t_cam_world]
    HmtxN = Hmtx / Hmtx[2, 2]  # Normalize
    h1, h2 = HmtxN[:, 0], HmtxN[:, 1]

    r1, r2 = h1, h2
    r3 = np.cross(r1, r2)  # Orthogonality of rotation matrix

    # Normalization
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = r3 / np.linalg.norm(r3)  # z-axis direction

    pos_cam_robot = pos_cam_marker - (ROBOT_HEIGHT * r3)

    # Projecting the robot base position to image plane
    uv_robot, _ = cv2.projectPoints(pos_cam_robot.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    uv_robot = np.array([[[uv_robot[0][0][0], uv_robot[0][0][1]]]], dtype=np.float32)

    # lambda * [Xh, Yh, 1].T = Hmtx * [u, v, 1].T
    pos_world_robot = cv2.perspectiveTransform(uv_robot, Hmtx).squeeze()
    xh, yh = pos_world_robot[0], pos_world_robot[1]

    return xh, yh


def compute_camera_pose_from_anchors(
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    corners: Optional[npt.NDArray[np.float32]],
    ids: Optional[npt.NDArray[np.float32]],
    marker_positions: dict,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Compute the camera pose from the anchor ArUco tags.

    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param corners: The detected marker corners
    @param ids: The detected marker ids
    @param marker_positions: The marker ids and their real-world positions
    """
    if ids is None:
        return None

    image_points: list = []  # [x', y']
    object_points: list = []  # [X, Y, Z]

    half_size = MARKER_SIZE / 2.0
    corners_3d_loc = np.array(
        [
            [-half_size, +half_size, 0],  # Top-left
            [+half_size, +half_size, 0],  # Top-right
            [+half_size, -half_size, 0],  # Bottom-right
            [-half_size, -half_size, 0],  # Bottom-left
        ],
        dtype=np.float32,
    )

    if ids is not None and corners is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_positions:
                marker_corners_2d = corners[i][0]

                x, y = marker_positions[marker_id]

                markers_corners_3d = corners_3d_loc + np.array([x, y, 0])

                for corner_2d, corner_3d in zip(marker_corners_2d, markers_corners_3d):
                    image_points.append(corner_2d)
                    object_points.append(corner_3d)

    if len(image_points) < 4:
        return None

    image_points_arr = np.array(image_points, dtype=np.float32)
    object_points_arr = np.array(object_points, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points_arr, image_points_arr, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUAREE
    )

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

    rvec_cam_world, tvec_cam_world = camera_pose
    Rmtx_cam_world, _ = cv2.Rodrigues(rvec_cam_world)

    pos_cam_marker = tvec[0][0]  # Marker position in camera frame [xc, yc, zc]

    z_world_in_cam = Rmtx_cam_world[:, 2]  # z-axis direction

    pos_cam_robot = pos_cam_marker - (ROBOT_HEIGHT * z_world_in_cam)

    pos_world_robot = Rmtx_cam_world.T @ (pos_cam_robot - tvec_cam_world.flatten())

    xp, yp = pos_world_robot[0], pos_world_robot[1]

    return xp, yp


def compute_anchor_error(
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    corners: Optional[npt.NDArray[np.float32]],
    ids: Optional[npt.NDArray[np.float32]],
    marker_positions: dict,
    camera_pose: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """
    Compute the error between the detected anchor positions and their real-world positions.

    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param corners: The detected marker corners
    @param ids: The detected marker ids
    @param marker_positions: The marker ids and their real-world positions
    @param camera_pose: The camera pose (rvec, tvec) in world frame
    """
    if camera_pose is None:
        return 0.0, 0.0

    if ids is None:
        return 0.0, 0.0

    errors_x = []
    errors_y = []

    if ids is not None and corners is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_positions:
                _, tvec, _ = aruco.estimatePoseSingleMarkers([corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs)

                pos_meas = pos_cam_to_world(tvec, camera_pose)
                if pos_meas is None:
                    continue

                x_mes, y_mes = pos_meas
                x_real, y_real = marker_positions[marker_id]

                errors_x.append(x_mes - x_real)
                errors_y.append(y_mes - y_real)

    if len(errors_x) == 0 or len(errors_y) == 0:
        return 0.0, 0.0

    return np.mean(errors_x), np.mean(errors_y)


def estimate_robot_pose(
    img: np.ndarray,
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    marker_positions: dict,
    Hmtx: npt.NDArray[np.float64],
) -> dict[int, tuple[float, float, float]]:
    """
    Estimate the robot pose based on ArUco tags.

    @param img: The input image
    @param camera_matrix: The intrinsic matrix
    @param dist_coeffs: The distortion coefficients
    @param marker_positions: The marker ids and their real-world positions
    @param Hmtx: The homography matrix
    """
    img, corners, ids = detect_aruco(img, camera_matrix, dist_coeffs, display=False)

    robot_pose: dict[int, tuple[float, float, float]] = {}

    if ids is None:
        return robot_pose

    camera_pose = compute_camera_pose_from_anchors(camera_matrix, dist_coeffs, corners, ids, marker_positions)
    x_err, y_err = compute_anchor_error(camera_matrix, dist_coeffs, corners, ids, marker_positions, camera_pose)

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_positions or Hmtx is None:  # Don't process the tags used for mapping
            continue

        marker_corners = corners[i]
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers([marker_corners], MARKER_SIZE, camera_matrix, dist_coeffs)

        # Getting position with homography method
        xh, yh = reproject_marker_pos_to_ground(tvec, camera_matrix, dist_coeffs, Hmtx)
        x, y = xh, yh

        # Getting position with PnP
        if camera_pose is not None:
            pos = pos_cam_to_world(tvec, camera_pose)
            if pos is not None:
                xp, yp = pos

                x_corr = xp + x_err
                y_corr = yp + y_err

                x = WEIGHT_HMTX * xh + WEIGHT_PNP * x_corr
                y = WEIGHT_HMTX * yh + WEIGHT_PNP * y_corr

        # Getting yaw orientation with angle-axis method
        Rmtx_cam_marker, _ = cv2.Rodrigues(rvec[0])
        theta_z = np.arctan2(Rmtx_cam_marker[1, 0], Rmtx_cam_marker[0, 0])

        print(
            f"Robot n°{marker_id} pose: "
            f"x = {x*M_TO_CM:.2f} cm; "
            f"y = {y*M_TO_CM:.2f} cm; "
            f"theta = {np.rad2deg(theta_z):.1f}°"
        )
        robot_pose[marker_id] = (x, y, theta_z)

    return robot_pose


def pose_to_img(robot_pose: dict) -> np.ndarray:
    """
    Display the robot pose into an image.

    @robot_pose: The dictionnary of robot poses
    """

    ppm_x = IMAGE_WIDTH / WORLD_SIZE[0]
    ppm_y = IMAGE_HEIGHT / WORLD_SIZE[1]
    px_res = (ppm_x + ppm_y) // 2

    matrix = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)  # RGB format

    if len(robot_pose) == 0:
        return matrix

    for marker_id, (x, y, theta_z) in robot_pose.items():
        px, py = int(x * ppm_x), int(y * ppm_y)
        py = IMAGE_HEIGHT - py  # Putting the origin axis to the bottom left

        if 1 <= marker_id <= 5:  # Blue team
            color = (255, 50, 0)
        elif 6 <= marker_id <= 10:  # Yellow team
            color = (0, 205, 255)
        else:
            continue
        neg_color = (255 - color[0], 255 - color[1], 255 - color[2])

        # Robot position
        cv2.circle(matrix, (px, py), int(ROBOT_RADIUS * px_res), color, -1)

        # Robot orientation
        theta_z = theta_z.item()
        endx = int(px + ROBOT_RADIUS * px_res * np.cos(theta_z))
        endy = int(py + ROBOT_RADIUS * px_res * np.sin(theta_z))
        cv2.line(matrix, (px, py), (endx, endy), neg_color, 10)

    return matrix
