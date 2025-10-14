cmake_minimum_required(VERSION 3.8)
project(my_camera_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出编译命令，方便IDE和代码分析工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 依赖项查找 ---
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
# [!!!] MODIFIED LINE: Explicitly find the DNN component of OpenCV [!!!]
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn)

# --- 1. 构建装甲板识别节点 (作为独立可执行文件) ---
ament_auto_add_executable(armor_detector_node
  src/armor_detector_node.cpp
  # [!!!] ADDED LINE: Add the new source file to the build [!!!]
  src/number_classifier.cpp
)
target_link_libraries(armor_detector_node
  ${OpenCV_LIBS}
)

# --- 2. 构建三合一相机节点 (作为独立可执行文件) ---
ament_auto_add_executable(multi_source_camera_node
  src/multi_source_camera_node.cpp
)
# 为三合一相机节点链接海康SDK
target_include_directories(multi_source_camera_node PUBLIC
  "/opt/MVS/include"
)
target_link_directories(multi_source_camera_node PUBLIC
  "/opt/MVS/lib/64"
)
target_link_libraries(multi_source_camera_node
  MvCameraControl
  ${OpenCV_LIBS}
)

# --- 测试、安装和打包 ---
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