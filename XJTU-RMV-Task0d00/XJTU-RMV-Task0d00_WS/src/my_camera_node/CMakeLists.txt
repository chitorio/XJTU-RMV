cmake_minimum_required(VERSION 3.8)
project(my_camera_node)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出编译命令，方便IDE和代码分析工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 依赖项查找 ---
# ament_cmake_auto 会根据 package.xml 自动查找大部分ROS依赖
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 我们需要手动找到 OpenCV，因为它不是一个ROS包
find_package(OpenCV REQUIRED)


# --- 1. 构建海康相机节点 (作为可组合组件) ---
# 这个部分基本保持你原来的结构
# 我们将它编译成一个共享库，以便它可以被加载
ament_auto_add_library(${PROJECT_NAME}_component SHARED
  src/hik_camera_node.cpp  # 注意：这里是你的相机代码源文件名
)

# 为相机组件链接海康SDK
target_include_directories(${PROJECT_NAME}_component PUBLIC
  "/opt/MVS/include"
)
target_link_directories(${PROJECT_NAME}_component PUBLIC
  "/opt/MVS/lib/64"
)
target_link_libraries(${PROJECT_NAME}_component
  # 这里列出所有需要的海康SDK库
  MvCameraControl
  # 根据你的原始文件，可能还有以下这些，如果需要就取消注释
  # FormatConversion
  # MediaProcess
  # MVRender
  # MvUsb3vTL
)

# 将这个库注册为一个ROS2组件，并创建一个可以独立运行它的可执行文件
rclcpp_components_register_node(${PROJECT_NAME}_component
  PLUGIN hik_camera::HikCameraNode
  EXECUTABLE hik_camera_node # 生成的可执行文件名，例如 ros2 run task hik_camera_node
)


# --- 2. 构建装甲板识别节点 (作为独立可执行文件) ---
# 使用 ament_auto_add_executable 来创建一个新的可执行文件
ament_auto_add_executable(armor_detector_node
  src/armor_detector_node.cpp
)
# 为这个新的可执行文件链接它需要的库 (OpenCV)
target_link_libraries(armor_detector_node
  ${OpenCV_LIBS}
)


# --- 3. 构建视频发布节点 (作为独立可执行文件) ---
# 同样，为视频发布节点创建一个可执行文件
ament_auto_add_executable(video_publisher_node
  src/video_publisher_node.cpp
)
# 也为它链接OpenCV库
target_link_libraries(video_publisher_node
  ${OpenCV_LIBS}
)


# --- 测试、安装和打包 ---
# 这部分保持不变，ament_cmake_auto 会自动处理新创建的可执行文件的安装
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)