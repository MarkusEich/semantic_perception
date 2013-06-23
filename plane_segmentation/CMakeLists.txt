cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(PCL 1.5 REQUIRED COMPONENTS segmentation filters)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

INCLUDE(FindPkgConfig)
pkg_check_modules(SEGMENTATION REQUIRED "segmentation")
INCLUDE_DIRECTORIES(${SEGMENTATION_INCLUDE_DIRS})
LINK_DIRECTORIES(${SEGMENTATION_LIBRARY_DIRS})

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
rosbuild_add_executable(plane_segmentation src/plane_segmentation.cpp)
target_link_libraries(${PROJECT_NAME} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_FILTERS_LIBRARIES} pcl_filters pcl_segmentation pcl_features ${SEGMENTATION_LIBRARIES})
