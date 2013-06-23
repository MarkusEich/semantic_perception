cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
INCLUDE( ${CMAKE_ROOT}/Modules/FindPkgConfig.cmake)

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

find_package(PCL 1.5 REQUIRED COMPONENTS segmentation filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
MESSAGE(STATUS "Linked to the libs :" ${PCL_LIBRARIES})

#Add the Boost Stuff
find_package(Boost 1.46 COMPONENTS system thread)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()
#include_directories(./FuzzyDLProducer/include)
#link_directories(${PROJECT_SOURCE_DIR}/lib)

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
rosbuild_add_executable(${PROJECT_NAME} src/cylinder_segmentation.cpp)
rosbuild_link_boost(${PROJECT_NAME} thread system)
target_link_libraries(${PROJECT_NAME}  ${PCL_LIBRARIES} pcl_features)
#${PCL_SEGMENTATION_LIBRARIES} ${PCL_FILTERS_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} )
#target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
