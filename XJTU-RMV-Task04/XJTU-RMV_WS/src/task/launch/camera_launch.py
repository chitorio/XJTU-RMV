import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('task'), 'config', 'camera_params.yaml'
    )

    camera_info_url = 'package://task/config/camera_info.yaml'

    return LaunchDescription([
        DeclareLaunchArgument(
            name='params_file',
            default_value=params_file
        ),
        DeclareLaunchArgument(
            name='camera_info_url',
            default_value=camera_info_url
        ),
        DeclareLaunchArgument(
            name='use_sensor_data_qos',
            default_value='false'
        ),

        Node(
            package='task',                         # 包名
            executable='task_node',                 # 可执行文件名（从 CMakeLists 来的）
            name='camera_node',                 # 节点名称
            output='screen',
            emulate_tty=True,
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'camera_info_url': LaunchConfiguration('camera_info_url')
                },
            ],
        )
    ])
