from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package',
            executable='video_publisher_node',
            name='video_publisher',
            parameters=[{
                'video_path': '/path/to/your/video.mp4',
            }]
        ),
        Node(
            package='your_package',
            executable='armor_detector_node',
            name='armor_detector',
            parameters=[{
                'debug': False,  # 关闭调试输出
                'min_contour_area': 20,
                'max_contour_area': 400,
                'use_morphology': True,
            }]
        )
    ])