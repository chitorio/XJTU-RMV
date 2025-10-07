# MVS SDK二次开发
***
这是基于海康威视官网的MVS C++ SDK开发的一个功能完善、性能稳定、易于使用的ROS2功能包。用户可以轻松地在项目中使用海康相机，获取图像数据并控制相机基础参数。

## 依赖项
***
- Ubuntu 22.04
- ROS2 Humble
- 海康 MVS SDK

## 依赖安装
***
### 系统依赖
```bash
sudo apt update
sudo apt install ros-${ROS_DISTRO}-camera-info-manager \
                 ros-${ROS_DISTRO}-image-transport
```

### 海康威视SDK
1. 从海康威视官网下载 MVS SDK
2. 安装 SDK（通常包含动态库和头文件）
3. 确保 `MvCameraControl.h` 和 `libMVCameraCtrl.so` 在系统路径中

## 编译安装
***
```bash
# 创建工作空间
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 克隆代码（假设代码在此目录）
git clone <your-repository>

# 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select task

# 配置环境
source install/setup.bash
```

## 使用方法
***
你可以在`config/camera_params.yaml`中修改曝光时间 (Exposure Time)，增益 (Gain)，帧率 (Frame Rate)，图像格式 (Pixel Format)等

通过IP连接或USB连接相机后，在工作环境运行命令
```bash
# 启动相机节点
ros2 launch task camera_launch.py
```

之后，你就可以看到节点运行成功的输出信息！