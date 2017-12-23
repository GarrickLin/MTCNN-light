cmake_minimum_required(VERSION 2.8)
project(main)

set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(D:/SDK/OpenBLAS-v0.2.14-Win64-int64/include)
set(libopenblas D:/SDK/OpenBLAS-v0.2.14-Win64-int64/lib/libopenblas.dll.a)
find_package(OpenCV REQUIRED)
include_directories(include)

# mtcnn
file(GLOB mtcnn_src src/mtcnn/*.cpp)
add_library(libmtcnn ${mtcnn_src})
target_link_libraries(libmtcnn ${OpenCV_LIBS} ${libopenblas})

# demo
add_executable(demo src/demo/pikaqiu.cpp)
target_link_libraries(demo libmtcnn ${OpenCV_LIBS} ${libopenblas})


# wrapper
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_INCLUDE_DIR D:/SDK/boost_1_59_0)
include(cmake/boost-python.cmake)
find_package(PythonLibs 2)

set(TEST_PYTHON_VERSION "2"
    CACHE STRING "Python version used on tests.")
set_property(CACHE TEST_PYTHON_VERSION PROPERTY STRINGS "3" "2")

if(TEST_PYTHON_VERSION EQUAL 2)
    find_package(PythonLibs 2)
else()
    find_package(PythonLibs 3)
endif()

set(numpy_include E:/ProgramData/Anaconda2/Lib/site-packages/numpy/core/include)
set(numpy_lib E:/ProgramData/Anaconda2/Lib/site-packages/numpy/core/lib/npymath.lib)

include_directories(src/wrapper
    ${OpenCV_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${numpy_include})

# Build np<=>opencv converter library
add_library(wrapper src/wrapper/np_opencv_converter.cpp 
                    src/wrapper/conversion.cpp)
target_link_libraries(wrapper ${OpenCV_LIBS} ${numpy_lib})

# mtcnn module
boost_python_module(mtcnn src/python/mtcnn_module.cpp)
target_link_libraries(mtcnn libmtcnn wrapper ${OpenCV_LIBS} ${numpy_lib} ${libopenblas})

