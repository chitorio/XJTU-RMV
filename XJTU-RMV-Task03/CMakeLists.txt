cmake_minimum_required(VERSION 3.5)
project(task03)

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(test src/test.cpp)
add_executable(main src/main.cpp)

target_link_libraries(test ${CERES_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(main ${CERES_LIBRARIES} ${OpenCV_LIBS})
