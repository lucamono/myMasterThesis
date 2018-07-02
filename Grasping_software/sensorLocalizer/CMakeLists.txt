cmake_minimum_required(VERSION 2.8)
Project (Sensor_localizer CXX)

# TO BE SET PROPERLY
#set(CMAKE_MODULE_PATH /home/marco/FlexSight/DatasetTools/Kinect2Interface/modules)
set(CMAKE_MODULE_PATH /home/luca/Scrivania/Grasping_software/modules)
set(CVEXT_MAIN_DIR /home/luca/cv_ext)

OPTION(ENABLE_OPENCL "Enable OpenCL support" OFF)
SET(ENABLE_OPENCL OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)


find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${FREENECT2_INCLUDE_DIRS})
include_directories(${CVEXT_MAIN_DIR}/lib/)
include_directories(${CVEXT_MAIN_DIR}/include/)

find_package(Freenect2 REQUIRED)

include_directories(${FREENECT2_INCLUDE_DIRS})

if(ENABLE_OPENCL)
  find_package(OpenCL)

  if(OpenCL_FOUND)
    add_definitions("-DHAVE_OPENCL")
  endif(OpenCL_FOUND)

endif(ENABLE_OPENCL)


#link to libraries
set(CV_EXT_LIBS_DIR ${CVEXT_MAIN_DIR}/lib)
find_library( CV_EXT_LIBS cv_ext ${CV_EXT_LIBS_DIR} )


include_directories(include)

add_executable(run_test test.cpp sensor_localizer.cpp)

target_link_libraries(run_test ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${GLFW_LIBRARIES} ${PCL_LIBRARIES} ${CV_EXT_LIBS})

message(${FREENECT2_INCLUDE_DIRS}" ______ " ${GLFW_INCLUDE_DIRS}" ______ " ${GLFW_LIBRARIES} " ________ " ${FREENECT2_LIBRARY} " ________ " ${OpenCV_LIBS})

