"""Aruco processing node for ROS 2 that estimates robot pose, and publishes relevant data."""

import rclpy  # type: ignore
from pose2d_msgs.msg import Pose2D  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # type: ignore
from sensor_msgs.msg import Image  # type: ignore

from .aruco_detection import *


class ArucoProcessing(Node):
    """Node handling ArUco marker detection and robot pose estimation."""

    def __init__(self) -> None:
        """Initialize the ArucoProcessing node."""
        super().__init__("aruco_processing")

        # Sensor QoS
        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)

        # Pose QoS
        pose_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.subscriber = self.create_subscription(Image, "camera_feed", self.image_callback, sensor_qos)
        self.subscriber  # prevent unused variable warning

        self.aruco_frame_publisher = self.create_publisher(Image, "aruco_frame", sensor_qos)

        self.robot_pose_publisher = self.create_publisher(Pose2D, "robot_pose", pose_qos)

        self.robot_pose_in_map_publisher = self.create_publisher(Image, "robot_pose_in_map", sensor_qos)

        self.Hmtx: Optional[npt.NDArray[np.float64]] = None

    def image_callback(self, msg) -> None:
        """Image processing callback function.

        @param msg: The received message
        """
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = data.reshape((msg.height, msg.width, 3))

            aruco_frame, _, ids = detect_aruco(frame, camera_matrix, dist_coeffs)
            aruco_msg = self.mat_to_image_msg(aruco_frame, msg.header)
            self.aruco_frame_publisher.publish(aruco_msg)

            if self.Hmtx is None and ids is not None and all(x in ids.flatten() for x in MARKER_POSITIONS):
                self.Hmtx = compute_homography(frame, camera_matrix, dist_coeffs, MARKER_POSITIONS)

            if self.Hmtx is not None:
                robot_pose = estimate_robot_pose(frame, camera_matrix, dist_coeffs, MARKER_POSITIONS, self.Hmtx)

                for marker_id, (x, y, theta_z) in robot_pose.items():
                    pose_msg = Pose2D()
                    pose_msg.marker_id = int(marker_id)

                    pose_msg.x = float(x * M_TO_CM)
                    pose_msg.y = float(y * M_TO_CM)
                    pose_msg.theta = float(theta_z)

                    self.robot_pose_publisher.publish(pose_msg)

                pose_frame = pose_to_img(robot_pose)
                pose_in_map_msg = self.mat_to_image_msg(pose_frame, msg.header)
                self.robot_pose_in_map_publisher.publish(pose_in_map_msg)

        except Exception as e:
            self.get_logger().error(f"Error during image processing: {e}")

    def mat_to_image_msg(self, frame, header) -> Image:
        """
        Converts a openCV matrix to a ROS 2 Image msg.

        @param frame: The provided frame
        @param header: The header of the msg
        """
        msg = Image()
        msg.header = header
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ArucoProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
