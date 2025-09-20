cmake_minimum_required(VERSION 3.10)
project(Task02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 指定OpenCV的路径
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Task02
    src/main.cpp
)

target_link_libraries(Task02 PRIVATE ${OpenCV_LIBS})