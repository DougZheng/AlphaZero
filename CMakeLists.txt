cmake_minimum_required(VERSION 3.8.0)
project(AlphaZero_Gomoku)

# torch
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# python
# find_package(PythonLibs)
# include_directories(${PYTHON_INCLUDE_PATH})

# 设置编译参数
set(CMAKE_CXX_FLAGS "-std=c++14 -pthread -O3")

# 包含源文件
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

# 设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 生成可执行文件
add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)