cmake_minimum_required(VERSION 3.8.0)
project(AlphaZero_Gomoku)

# 设置编译参数
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -O3")

# 包含源文件
include_directories(./src)
aux_source_directory(./src DIR_SRCS)

# 添加头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 生成可执行文件
add_executable(${PROJECT_NAME} ${DIR_SRCS})