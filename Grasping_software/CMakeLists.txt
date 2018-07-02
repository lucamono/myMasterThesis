cmake_minimum_required(VERSION 2.8)
Project (Grasping_software CXX)

# TO BE SET PROPERLY
#set(CMAKE_MODULE_PATH /home/marco/FlexSight/DatasetTools/Kinect2Interface/modules)
set(CMAKE_MODULE_PATH /home/luca/Scrivania/Grasping_software/modules)
set(CVEXT_MAIN_DIR /home/luca/cv_ext)

OPTION(ENABLE_OPENCL "Enable OpenCL support" OFF)
SET(ENABLE_OPENCL OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
pkg_search_module(GLFW3 REQUIRED glfw3)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${GLFW3_INCLUDE_DIRS})
include_directories(${FREENECT2_INCLUDE_DIRS})
include_directories(${CVEXT_MAIN_DIR}/lib/)
include_directories(${CVEXT_MAIN_DIR}/include/)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(sensorLocalizer/include)

find_package(Freenect2 REQUIRED)

include_directories(${FREENECT2_INCLUDE_DIRS})

if(ENABLE_OPENCL)
  find_package(OpenCL)

  if(OpenCL_FOUND)
    add_definitions("-DHAVE_OPENCL")
  endif(OpenCL_FOUND)

endif(ENABLE_OPENCL)

find_library(DARKNETLIB darknet PATHS ${PROJECT_SOURCE_DIR}/darknet)
set(DARKNET_INCLUDE_DIR /home/luca/Scrivania/Grasping_software/darknet/include)
include_directories(${DARKNET_INCLUDE_DIR} include)

#link to libraries
set(CV_EXT_LIBS_DIR ${CVEXT_MAIN_DIR}/lib)
find_library( CV_EXT_LIBS cv_ext ${CV_EXT_LIBS_DIR} )

add_executable(GraspingExecutable grasping_software.cpp src/imageProcessing.cpp src/utils.cpp src/yoloDetector.cpp src/datasetInterface.cpp src/mySensorLocalizer.cpp sensorLocalizer/sensor_localizer.cpp
src/serial_communication.cpp)

target_link_libraries(GraspingExecutable ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${GLFW_LIBRARIES} ${PCL_LIBRARIES} ${DARKNETLIB} ${CV_EXT_LIBS})

message(${FREENECT2_INCLUDE_DIRS}" ______ " ${GLFW_INCLUDE_DIRS}" ______ " ${GLFW_LIBRARIES} " ________ " ${FREENECT2_LIBRARY} " ________ " ${OpenCV_LIBS})

