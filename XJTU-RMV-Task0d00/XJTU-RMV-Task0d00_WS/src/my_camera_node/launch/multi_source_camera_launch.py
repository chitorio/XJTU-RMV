# multi_source_camera_launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # --- 1. 获取配置文件的路径 ---
    # 这会找到您的 my_camera_node 包的安装路径
    pkg_share = get_package_share_directory('my_camera_node')
    
    # 拼接相机参数文件的完整路径
    camera_params_file = os.path.join(
        pkg_share,
        'config',
        'camera_params.yaml'
    )

    # [!!!] 新增：拼接模型文件的路径 [!!!]
    model_path_file = os.path.join(
        pkg_share,
        'model',
        'mlp.onnx'
    )
    label_path_file = os.path.join(
        pkg_share,
        'model',
        'labels.txt'
    )


    # --- 2. 声明可以在命令行中覆盖的启动参数 ---
    # 这是最关键的参数，用于切换模式
    source_mode_arg = DeclareLaunchArgument(
        'source_mode',
        default_value='hik_camera', # 默认使用海康相机
        description="Source mode: 'hik_camera' or 'video_file'"
    )

    # 视频文件路径参数
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='/home/zoot/Downloads/blue.mp4', # 您的视频文件路径
        description='Path to the video file'
    )

    # --- 3. 创建节点 ---
    
    # 相机节点
    camera_node = Node(
        package="my_camera_node",
        executable="multi_source_camera_node",
        name="multi_source_camera_node",
        output="screen",
        # 关键改动：
        # 1. 首先加载YAML文件中的所有参数
        # 2. 然后，用一个字典来覆盖在命令行中指定的参数
        parameters=[
            camera_params_file,
            {
                'source_mode': LaunchConfiguration('source_mode'),
                'video_path': LaunchConfiguration('video_path'),
            }
        ]
    )

    # [!!!] 修改点：为装甲板检测节点添加模型参数 [!!!]
    detector_node = Node(
        package="my_camera_node",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[{
            # --- 原有的参数 ---
            "binary_thres": 200,
            "detect_color": 1, # 0=RED, 1=BLUE
            "debug": True,
            
            # --- 新增的模型参数 ---
            "model_path": model_path_file,
            "label_path": label_path_file,
            "classifier_threshold": 0.75, # 分类器置信度阈值，您可以根据需要调整

            # --- 其他参数也可以在这里设置 ---
            # "pnp_update_rate": 3, 
            # "armor_width": 0.135,
        }],

        # 调试日志
        arguments=['--ros-args', '--log-level', 'info']
    )

    return LaunchDescription([
        # 必须先声明启动参数
        source_mode_arg,
        video_path_arg,
        
        # 然后是节点
        camera_node,
        detector_node
    ])