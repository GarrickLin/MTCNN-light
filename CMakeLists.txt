cmake_minimum_required(VERSION 2.8)
project(main)

set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(D:/SDK/OpenBLAS-v0.2.14-Win64-int64/include)
set(libopenblas D:/SDK/OpenBLAS-v0.2.14-Win64-int64/lib/libopenblas.dll.a)

find_package( OpenCV REQUIRED)

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable(main ${DIR_SRCS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${libopenblas})
