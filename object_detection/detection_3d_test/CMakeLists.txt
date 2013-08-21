PROJECT(detection_3D_test)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
include (FindPkgConfig)

if (PKG_CONFIG_FOUND)
  pkg_check_modules(DETECTION object_detection)
endif()
include_directories(${DETECTION_INCLUDE_DIRS})
link_directories(${DETECTION_LIBRARY_DIRS})

find_package(PCL 1.7 REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Boost 1.46 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

MESSAGE(STATUS "Needed libs: " ${DETECTION_LIBRARIES})

add_executable(detection_3D_test src/detection_3D_test.cpp)
target_link_libraries(detection_3D_test ${DETECTION_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
install (TARGETS detection_3D_test DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
