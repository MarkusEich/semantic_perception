#include "BlobDetection.hpp"
#include <boost/foreach.hpp>

using namespace object_detection;

void BlobDetection::setConfiguration( const Configuration& config )
{
    this->config = config;
}

void BlobDetection::setObjectList( const std::vector<PrimitiveObject> &list )
{
    objList.clear();
    BOOST_FOREACH( const PrimitiveObject &p, list )
    {
	objList.push_back( p.getObject() );
    }
}

void BlobDetection::processImage( const cv::Mat &frame )
{
    candidates.resize( objList.size() );

    dbg_img = frame.clone();
    // do the processing here
    // and store blob candidates in candidates structure

}

cv::Mat BlobDetection::getDebugImage()
{
    // TODO
    // take the input image and paint the candidates as blobs 
    return dbg_img;
}

std::vector<std::vector<Blob> > BlobDetection::getCandidates() const
{
    assert( candidates.size() == objList.size() ); 
    return candidates;
}
