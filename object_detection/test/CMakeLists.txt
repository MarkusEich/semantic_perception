rock_testsuite(test suite.cpp
    test_BlobDetect.cpp
    test_3D_Detection.cpp
    DEPS object_detection 
    DEPS_CMAKE PCL 
)
