from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ===== 相机节点参数 =====
    source_mode_arg = DeclareLaunchArgument(
        "source_mode",
        default_value="hik_camera",
        description="视频源模式: 'hik_camera' 或 'video_file'"
    )

    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value="/home/zoot/Downloads/blue.mp4",
        description="视频文件路径（video_file模式使用）"
    )

    # 海康相机参数
    camera_ip_arg = DeclareLaunchArgument(
        "camera_ip",
        default_value="",
        description="海康相机IP地址（hik_camera模式使用）"
    )

    camera_serial_arg = DeclareLaunchArgument(
        "camera_serial",
        default_value="",
        description="海康相机序列号（hik_camera模式使用）"
    )

    # 通用参数
    frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="120.0",
        description="帧率设置"
    )

    loop_video_arg = DeclareLaunchArgument(
        "loop_video",
        default_value="true",
        description="是否循环播放视频（video_file模式）"
    )

    resize_arg = DeclareLaunchArgument(
        "resize",
        default_value="true",
        description="是否启用图像缩放"
    )

    resize_width_arg = DeclareLaunchArgument(
        "resize_width",
        default_value="640",
        description="缩放后的宽度"
    )

    resize_height_arg = DeclareLaunchArgument(
        "resize_height",
        default_value="480",
        description="缩放后的高度"
    )

    # 海康相机控制参数
    exposure_time_arg = DeclareLaunchArgument(
        "exposure_time",
        default_value="8000.0",
        description="曝光时间（hik_camera模式）"
    )

    gain_arg = DeclareLaunchArgument(
        "gain",
        default_value="10.0",
        description="增益（hik_camera模式）"
    )

    # ===== 装甲板检测节点参数 =====
    detector_thres_arg = DeclareLaunchArgument(
        "detector_thres",
        default_value="160",
        description="二值化阈值 (0-255)"
    )

    detect_color_arg = DeclareLaunchArgument(
        "detect_color", 
        default_value="1",
        description="检测颜色: 0=红色, 1=蓝色"
    )

    debug_mode_arg = DeclareLaunchArgument(
        "debug_mode",
        default_value="true",
        description="是否开启调试模式"
    )

    # ===== 创建节点 =====
    camera_node = Node(
        package="my_camera_node",
        executable="multi_source_camera_node",
        name="multi_source_camera_node",
        output="screen",
        parameters=[{
            "source_mode": LaunchConfiguration("source_mode"),
            "video_path": LaunchConfiguration("video_path"),
            "camera_ip": LaunchConfiguration("camera_ip"),
            "camera_serial": LaunchConfiguration("camera_serial"),
            "frame_rate": LaunchConfiguration("frame_rate"),
            "loop_video": LaunchConfiguration("loop_video"),
            "resize": LaunchConfiguration("resize"),
            "resize_width": LaunchConfiguration("resize_width"),
            "resize_height": LaunchConfiguration("resize_height"),
            "exposure_time": LaunchConfiguration("exposure_time"),
            "gain": LaunchConfiguration("gain"),
        }]
    )

    detector_node = Node(
        package="my_camera_node",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[{
            "binary_thres": LaunchConfiguration("detector_thres"),
            "detect_color": LaunchConfiguration("detect_color"),
            "debug": LaunchConfiguration("debug_mode"),
        }]
    )

    return LaunchDescription([
        # 相机参数
        source_mode_arg,
        video_path_arg,
        camera_ip_arg,
        camera_serial_arg,
        frame_rate_arg,
        loop_video_arg,
        resize_arg,
        resize_width_arg,
        resize_height_arg,
        exposure_time_arg,
        gain_arg,
        
        # 检测器参数
        detector_thres_arg,
        detect_color_arg,
        debug_mode_arg,
        
        # 节点
        camera_node,
        detector_node
    ])