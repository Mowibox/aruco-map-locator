import socket
from launch import LaunchDescription
from launch_ros.actions import Node


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def generate_launch_description():
    IP_ADRESS = get_host_ip()

    return LaunchDescription([
        Node(
            package='image_provider',
            executable='image_subscriber',
            name='image_subscriber',
        ),

        Node(
            package='aruco_map_locator',
            executable='aruco_processing',
            name='aruco_processing',
        ),

        Node(
            package='image_provider',
            executable='image_subscriber',
            name='image_subscriber_1',
            parameters=[{'topic': 'aruco_frame'}]
        ),

        Node(
            package='image_provider',
            executable='image_subscriber',
            name='image_subscriber_2',
            output='screen',
            parameters=[{'topic': 'robot_pose_in_map'}]
        ),

        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            name='tcp_server',
            parameters=[{'ROS_IP': IP_ADRESS}]
        )
    ])