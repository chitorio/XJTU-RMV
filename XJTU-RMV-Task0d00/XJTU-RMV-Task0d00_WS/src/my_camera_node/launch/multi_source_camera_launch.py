from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ===== 相机节点参数 =====
    source_mode_arg = DeclareLaunchArgument(
        "source_mode",
        default_value="usb_camera",
        description="视频源模式: 'video_file' 或 'usb_camera'"
    )

    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value="/home/zoot/Downloads/blue.mp4",
        description="视频文件路径"
    )

    camera_id_arg = DeclareLaunchArgument(
        "camera_id",
        default_value="0",
        description="USB 相机 ID"
    )

    frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="60.0",
        description="帧率设置"
    )

    loop_video_arg = DeclareLaunchArgument(
        "loop_video",
        default_value="true",
        description="是否循环播放视频"
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

    use_thread_arg = DeclareLaunchArgument(
        "use_thread",
        default_value="true",
        description="是否使用多线程模式（更高性能）"
    )

    use_hw_accel_arg = DeclareLaunchArgument(
        "use_hw_accel", 
        default_value="true",
        description="是否启用硬件加速"
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
        description="是否开启调试模式（显示更多信息）"
    )

    min_contour_area_arg = DeclareLaunchArgument(
        "min_contour_area",
        default_value="50",
        description="最小轮廓面积"
    )

    max_contour_area_arg = DeclareLaunchArgument(
        "max_contour_area",
        default_value="3000", 
        description="最大轮廓面积"
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
            "camera_id": LaunchConfiguration("camera_id"),
            "frame_rate": LaunchConfiguration("frame_rate"),
            "loop_video": LaunchConfiguration("loop_video"),
            "resize": LaunchConfiguration("resize"),
            "resize_width": LaunchConfiguration("resize_width"),
            "resize_height": LaunchConfiguration("resize_height"),
            "use_thread": LaunchConfiguration("use_thread"),
            "use_hw_accel": LaunchConfiguration("use_hw_accel"),
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
            "min_contour_area": LaunchConfiguration("min_contour_area"),
            "max_contour_area": LaunchConfiguration("max_contour_area"),
        }]
    )

    return LaunchDescription([
        # 相机参数
        source_mode_arg,
        video_path_arg,
        camera_id_arg,
        frame_rate_arg,
        loop_video_arg,
        resize_arg,
        resize_width_arg,
        resize_height_arg,
        use_thread_arg,
        use_hw_accel_arg,
        
        # 检测器参数
        detector_thres_arg,
        detect_color_arg,
        debug_mode_arg,
        min_contour_area_arg,
        max_contour_area_arg,
        
        # 节点
        camera_node,
        detector_node
    ])