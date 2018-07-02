project( flexsight_core )

cmake_minimum_required (VERSION 2.8) 
cmake_policy(SET CMP0015 NEW)


if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -std=c++11 -march=native -D LINUX")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -std=c++11 -march=native -O3 -D LINUX")

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package(PCL 1.6 REQUIRED )
add_definitions(${PCL_DEFINITIONS})
find_package( Boost COMPONENTS program_options filesystem system REQUIRED )
find_package(Ceres  REQUIRED)


add_subdirectory(${PROJECT_SOURCE_DIR}/externals/cv_ext)
add_subdirectory(${PROJECT_SOURCE_DIR}/externals/d2co)

include_directories( ./include
                     ./apps
                     ./externals/spinnaker_sdk/include
                     ./externals/d2co/include
                     /usr/include
                     /usr/local/include
                     ${CV_EXT_INCLUDE_DIR}
                     ${PCL_INCLUDE_DIRS}
                     ${OpenCV_INCLUDE_DIRS})

find_library( SPINNAKER_LIB Spinnaker ./externals/spinnaker_sdk/lib )
find_library( OBJECT_RECOGNTION_LIB object_recogntion ./externals/d2co/lib )

aux_source_directory( src FLEXSIGHT_CORE_SRC )

add_library(flexsight_core STATIC ${FLEXSIGHT_CORE_SRC})
set_target_properties(flexsight_core PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_executable(test_acq apps/test_acq.cpp apps/camera_utils.cpp)
target_link_libraries( test_acq
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_acq PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_acq_thread apps/test_acq_thread.cpp apps/camera_utils.cpp)
target_link_libraries( test_acq_thread
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_acq_thread PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_software_trigger apps/test_software_trigger.cpp apps/camera_utils.cpp)
target_link_libraries( test_software_trigger
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_software_trigger PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_hardware_trigger apps/test_hardware_trigger.cpp apps/camera_utils.cpp)
target_link_libraries( test_hardware_trigger
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_hardware_trigger PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_acq_sync apps/test_acq_sync.cpp apps/camera_utils.cpp)
target_link_libraries( test_acq_sync
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_acq_sync PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_camera_calib apps/test_camera_calib.cpp )
target_link_libraries( test_camera_calib
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES})
set_target_properties( test_camera_calib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(find_extrinsics apps/find_extrinsics.cpp )
target_link_libraries( find_extrinsics
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES})
set_target_properties( find_extrinsics PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_acq_hdr apps/test_acq_hdr.cpp apps/camera_utils.cpp)
target_link_libraries( test_acq_hdr
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS})
set_target_properties( test_acq_hdr PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )


# include_directories($ENV{HALCONROOT}/include $ENV{HALCONROOT}/include/halconcpp $ENV{HALCONROOT}/include/hdevengine)
# link_directories(/home/freelist/Libraries/HalconLib/lib/x64-linux)
# add_executable(demo apps/demo.cpp apps/camera_utils.cpp)
# target_link_libraries( demo
#                        flexsight_core
#                        ${SPINNAKER_LIB}
#                        ${CV_EXT_LIBS}
#                        ${OpenCV_LIBS}
#                        $ENV{HALCONROOT}/lib/$ENV{HALCONARCH}/libhdevenginecpp.so
#                        $ENV{HALCONROOT}/lib/$ENV{HALCONARCH}/libhalconcpp.so
#                        $ENV{HALCONROOT}/lib/$ENV{HALCONARCH}/libhalcon.so)
# set_target_properties( demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_object_detection apps/test_object_detection.cpp )
target_link_libraries( test_object_detection
                       flexsight_core
                       ${OBJECT_RECOGNTION_LIB}
                       ${SPINNAKER_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES}
                       ${DIME_LIBS}
                       ${PCL_LIBRARIES}
                       ${CERES_LIBRARIES}
                       ${OPENGL_LIBRARY}
                       ${GLEW_LIBS}
                       ${GLFW_STATIC_LIBRARIES}
                       ${OPENMESHCORE_LIBS})
set_target_properties( test_object_detection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

add_executable(test_online_object_detection apps/test_online_object_detection.cpp  apps/camera_utils.cpp)
target_link_libraries( test_online_object_detection
                       flexsight_core
                       ${SPINNAKER_LIB}
                       ${OBJECT_RECOGNTION_LIB}
                       ${CV_EXT_LIBS}
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES}
                       ${DIME_LIBS}
                       ${PCL_LIBRARIES}
                       ${CERES_LIBRARIES}
                       ${OPENGL_LIBRARY}
                       ${GLEW_LIBS}
                       ${GLFW_STATIC_LIBRARIES}
                       ${OPENMESHCORE_LIBS})
set_target_properties( test_online_object_detection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

#Obsolte apps
#
# add_executable(test_acquisition apps/test_acquisition.cpp)
# target_link_libraries( test_acquisition
#                        ${SPINNAKER_LIB}
#                        ${CV_EXT_LIBS}
#                        ${OpenCV_LIBS})
# set_target_properties( test_acquisition PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )
#
# add_executable(test_trigger apps/test_trigger.cpp)
# target_link_libraries( test_trigger
#                        ${SPINNAKER_LIB}
#                        ${CV_EXT_LIBS}
#                        ${OpenCV_LIBS})
# set_target_properties( test_trigger PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )
