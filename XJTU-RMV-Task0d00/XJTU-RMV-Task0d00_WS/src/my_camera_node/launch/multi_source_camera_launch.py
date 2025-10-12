from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 定义可配置参数
    source_mode_arg = DeclareLaunchArgument(
        "source_mode",
        default_value="video_file",
        description="视频源模式: 'video_file' 或 'usb_camera'"
    )

    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value="/home/zoot/Desktop/test.mp4",  # ⚙️ 修改成你的视频路径
        description="视频文件路径"
    )

    camera_id_arg = DeclareLaunchArgument(
        "camera_id",
        default_value="0",
        description="USB 相机 ID"
    )

    frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="30.0",
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

    # 创建节点
    camera_node = Node(
        package="my_camera_node",  # ⚙️ 改成你包的名字
        executable="multi_source_camera_node",  # 你编译后的可执行文件名
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
        }]
    )

    return LaunchDescription([
        source_mode_arg,
        video_path_arg,
        camera_id_arg,
        frame_rate_arg,
        loop_video_arg,
        resize_arg,
        resize_width_arg,
        resize_height_arg,
        camera_node
    ])
