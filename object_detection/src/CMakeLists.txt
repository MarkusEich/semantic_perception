rock_library(object_detection
    SOURCES
        BlobDetection.cpp
        PointCloudDetection.cpp        
    HEADERS 
	Configuration.hpp
	BlobDetection.hpp
        PointCloudDetection.hpp
	Objects.hpp        
    DEPS_PKGCONFIG base-types opencv segmentation    
    #This is a problem if multiple versions are installed
    #should take a defined version
    #DEPS_CMAKE PCL 
    )
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(object_detection ${PCL_LIBRARIES})
