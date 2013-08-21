#ifndef OBJECT_DETECTION_BLOB_DETECTION_HPP__
#define OBJECT_DETECTION_BLOB_DETECTION_HPP__

#include <object_detection/Configuration.hpp>
#include <object_detection/Objects.hpp>
#include <opencv2/opencv.hpp>

namespace object_detection
{

struct Blob
{
    /** 
     * diameter of the blob normalized for distance. 
     *
     * So that for distance to blob in m:
     * actual_diameter = diameter * distance
     */ 
    float diameter;

    /**
     * value between 0 and 1, where 0 is no confidence, and 1 is absolutely sure
     */
    float confidence;
};

class BlobDetection
{ 
    Configuration config;
    std::vector<Object::Ptr> objList; 
    std::vector<std::vector<Blob> > candidates;

    cv::Mat dbg_img;

public: 
    /**
     * set the configuration for the blob detection
     */
    void setConfiguration( const Configuration& config );

    /**
     * set the objects which should be detected
     *
     * note: this is quite focused detection of 3d objects with a known size
     * and color. There is no reason not to include some sort of intermediate
     * representation of what we are actually looking for.
     */
    void setObjectList( const std::vector<PrimitiveObject> &list );

    /**
     * Process the image frame, and try to find the objects in this frame.
     *
     * The result is returned using the getCandidates method
     *
     */
    void processImage( const cv::Mat &frame );

    /** 
     * generate a debug image, which shows the internal state of the processing
     * in a human understandable format
     */
    cv::Mat getDebugImage();

    /**
     * @brief return the number of candidate blobs which have been found in the image
     *
     * Vector will be empty if no adequate blobs have been found, or if
     * processImage() has not been called yet.
     *
     * @return a vector of vector of candidate blobs that have been found
     */
    std::vector<std::vector<Blob> > getCandidates() const;
};

} // end namespace object_detection

#endif 
