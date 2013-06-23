cmake_minimum_required(VERSION 2.4 FATAL_ERROR)

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT (segmentation)
SET(PROJECT_VERSION 1.0)

SET(CMAKE_CXX_FLAGS "-std=c++0x")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frounding-math -O3 -fopenmp -std=gnu++0x)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -O3 -fopenmp -std=gnu++0x)


FIND_LIBRARY(LIB_GSL gsl /usr/lib REQUIRED)
FIND_LIBRARY(LIB_GSLCBLAS gslcblas /usr/lib REQUIRED)
FIND_LIBRARY(LIB_DB db_cxx /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu REQUIRED)
FIND_PACKAGE(Boost COMPONENTS thread regex REQUIRED)
FIND_PACKAGE(CGAL COMPONENTS Core)
find_package(OpenCV REQUIRED )


INCLUDE(FindPkgConfig)


OPTION(USE_DEBUG "Set Debug points for vizualization" false)
if(USE_DEBUG)
ADD_DEFINITIONS(-DUSE_DEBUG)
endif()


INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#FIND_PACKAGE(CGAL)
if(CGAL_FOUND)
	INCLUDE(${CGAL_USE_FILE})
endif()

ADD_DEFINITIONS(
	-DTRANSPARENCY=64
	-std=gnu++0x
)


FIND_PATH(
	EIGEN3_INCLUDE_DIR 
	Eigen/Core 
	${CMAKE_INSTALL_PREFIX}/include/eigen3
	/usr/include/eigen3 
	/usr/local/include/eigen3 
REQUIRED)

MESSAGE(STATUS "GSL_LIBRARY PATH:         "${LIB_GSL} )
MESSAGE(STATUS "GSL_CBLAS_LIBRARY PATH:         "${LIB_GSLCBLAS} )
MESSAGE(STATUS "CGAL_LIBRARY PATH:         "${CGAL_LIBRARY} )
MESSAGE(STATUS "OPEN_CV_LIBS:         "${OpenCV_LIBS})

INCLUDE_DIRECTORIES(
	${CGAL_INCLUDE_DIR} 
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	/usr/include/db4.8/
	.
 )

MESSAGE("EIGEN INCLUDE DIR: ${EIGEN3_INCLUDE_DIR}")

find_package(CGAL QUIET COMPONENTS Core)

if ( CGAL_FOUND )
  include( ${CGAL_USE_FILE} )
else() 
    message(STATUS "This program requires the CGAL library, and will not be compiled.")  
endif()



if(NOT CGAL_FOUND)
TARGET_LINK_LIBRARIES(segmentationApp
	CGAL	
	CGALcore++
)
endif()

ADD_LIBRARY(segmentation SHARED
	regionGrowing.cpp
	kdtree.cpp
	plane.cpp
	hull.cpp
	spatialObject.cpp
	)


TARGET_LINK_LIBRARIES(segmentation
	${LIB_GSL}
	${LIB_GSLCBLAS}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
	${LIB_DB}

)

if(NOT CGAL_FOUND)
TARGET_LINK_LIBRARIES(segmentation
	CGAL	
	CGALcore++
)
endif()

CONFIGURE_FILE(segmentation.pc.in segmentation.pc @ONLY)


INSTALL(TARGETS segmentation 
        LIBRARY DESTINATION lib
	)


INSTALL(FILES 
	regionGrowing.h
	hull.h
	point.h
	rotatedRectangle.h
	kdtree.hpp
	plane.h
	spatialObject.h
	point3d.h
DESTINATION include)

INSTALL(FILES 
	${CMAKE_BINARY_DIR}/segmentation.pc 
DESTINATION lib/pkgconfig
								                                                )
	

