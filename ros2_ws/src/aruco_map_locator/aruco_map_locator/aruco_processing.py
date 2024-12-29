import rclpy
from rclpy.node import Node
from .aruco_detection import *
from sensor_msgs.msg import Image

class ArucoProcessing(Node):

    def __init__(self):
        super().__init__('aruco_processing')
        self.subscriber = self.create_subscription(
            Image, 'camera_feed', self.image_callback, 20)
        self.subscriber  # prevent unused variable warning

        self.aruco_frame_publisher = self.create_publisher(
            Image, 'aruco_frame', 20
        )

        self.homographic_projection_publisher = self.create_publisher(
            Image, 'homographic_projection', 20
        )

        self.robot_pose_in_map_publisher = self.create_publisher(
            Image, 'robot_pose_in_map', 20
        )

    def image_callback(self, msg):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = data.reshape((msg.height, msg.width, 3))

            aruco_frame, _, _ = detect_aruco(frame, camera_matrix, dist_coeffs)
            aruco_msg = self.mat_to_image_msg(aruco_frame, msg.header)

            h_frame, hmtx = compute_homography(frame, camera_matrix, dist_coeffs, marker_positions)
            h_msg = self.mat_to_image_msg(h_frame, msg.header)

            robot_pose = estimate_robot_pose(frame, camera_matrix, dist_coeffs, marker_positions, hmtx)
            pose_frame = pose_to_img(robot_pose)
            pose_msg = self.mat_to_image_msg(pose_frame, msg.header)

            self.aruco_frame_publisher.publish(aruco_msg)
            self.homographic_projection_publisher.publish(h_msg)
            self.robot_pose_in_map_publisher.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error during image processing: {e}")
    

    def mat_to_image_msg(self, frame, header):
        """
        Converts a openCV matrix to a ROS 2 Image msg

        @param frame: The provided frame
        @param header: The header of the msg
        """
        msg = Image()
        msg.header = header
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = frame.shape[1]*3
        msg.data = frame.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)

    node = ArucoProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()