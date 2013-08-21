PROJECT(ros_detection_test)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
include (FindPkgConfig)

if (PKG_CONFIG_FOUND)
  pkg_check_modules(DETECTION object_detection)
endif()
include_directories(${DETECTION_INCLUDE_DIRS})
link_directories(${DETECTION_LIBRARY_DIRS})

include(cmake/rospack.cmake)

get_rospack_flags(roscpp)
include_directories(${roscpp_INCLUDE_DIRS})
link_directories(${roscpp_LIBRARY_DIRS})

get_rospack_flags(tf)
include_directories(${tf_INCLUDE_DIRS})
link_directories(${tf_LIBRARY_DIRS} )

get_rospack_flags(pcl_conversions)
include_directories(${pcl_conversions_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED COMPONENTS segmentation filters io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ros_detection_test src/ros_detection_test.cpp)
target_link_libraries(ros_detection_test ${DETECTION_LIBRARIES} ${roscpp_LIBRARIES} ${tf_LIBRARIES} ${PCL_LIBRARIES})
install (TARGETS ros_detection_test DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
