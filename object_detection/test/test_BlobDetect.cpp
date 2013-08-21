#include <boost/test/unit_test.hpp>
#include <object_detection/BlobDetection.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace object_detection;

BOOST_AUTO_TEST_CASE( detect_blob_in_image )
{
    // load a test image and see 
    // if the right blobs are detected
    
    cv::Mat cv_image = cv::imread( "./test/test1.png" );

    std::vector<PrimitiveObject> objs;
    Cylinder cyl;
    // TODO update test values
    cyl.color = Color( 1.0, 0, 0 );
    cyl.diameter = 0.10;
    cyl.height = 0.20;
    objs.push_back( PrimitiveObject( cyl ) );
    Box box;
    box.color = Color( 1.0, 1.0, 0 );
    box.dimensions = base::Vector3d( 0.1, 0.05, 0.2 );

    BlobDetection blob;
    blob.setObjectList( objs );

    blob.processImage( cv_image );

    // TODO
    // do some automated test here
    // for some criteria
    // e.g. no more than 3 candidates per blob, and the blob with the highest
    // rating should be within 1% of the true position or so

    cv::imshow( "Debug Image", blob.getDebugImage() );
    cv::waitKey(0);
}
