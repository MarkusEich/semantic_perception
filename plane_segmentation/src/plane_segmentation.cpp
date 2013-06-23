#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl/ModelCoefficients.h>

#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <stdlib.h>
#include <pcl/filters/extract_indices.h>
#include "pcl/filters/project_inliers.h"
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>


#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <impera_shape_msgs/PlaneBag.h>

#include <plane.h>
#include <fstream>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointRGBT;

#define RANSAC_THRESHOLD 3000

#define RUN 1
#define STOP 0
#define NOT_SET "NOT_SET"

std::string nodeName="plane_segmentation";

class PlaneSegmentation{

public:

  PlaneSegmentation(ros::NodeHandle &n):_n(n){

    ros::param::param("~ransac_iterations",(int&) _ransac_iterations, 1000);
    ros::param::param("~ransac_dist_thresold",(double&) _ransac_dist_threshold, 0.05);
    ros::param::param("~ransac_normal_dist_weight",(double&) _ransac_normal_dist_weight, 0.1);
    ros::param::param("~segmentation_cluster_dist",(double&) _segmentation_cluster_dist, 0.1);
    ros::param::param("~outlier_radius",(double&) _outlier_radius,0.3);
    ros::param::param("~outlier_nn",(int&) _outlier_nn,20);
    ros::param::param("~subsample",(double&) subsample_leaf_size,0.02);
    ros::param::param("~min_cluster_size",(int&) min_cluster_size_,500);


    _sub=_n.subscribe("/cloud",1,&PlaneSegmentation::scan_callback,this);
    _pub_plane=_n.advertise<sensor_msgs::PointCloud2> ("planes",10);
    _pub_plane_bag=_n.advertise<impera_shape_msgs::PlaneBag> ("plane_bag",2);
    _pub_residuals=_n.advertise<sensor_msgs::PointCloud2> ("plane_residuals",10);

    srand (time(NULL));//random seed for random colors
  }
 
  void scan_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_ptr){    


    pcl::PointCloud<PointRGBT> input_cloud;// the raw input point

    pcl::ModelCoefficients coefficients_plane; //the plane coefficents generated by segmentation
    pcl::PointIndices inliers_plane;//the indices of points which lay in the plane        

    //Point cloud for storing the cloud_normals
    pcl::PointCloud<pcl::Normal> cloud_normals;

    //all points within a plane, use a color identifier
    pcl::PointCloud<PointRGBT> cloud_plane;

    //for publishing
    sensor_msgs::PointCloud2 cloud_plane_msg;
    sensor_msgs::PointCloud2 residual_msg;
    impera_shape_msgs::PlaneBag plane_bag_msg;

    pcl::fromROSMsg(*cloud_ptr, input_cloud);    
    
    ROS_INFO("Received %d points for plane segmentation",(int)input_cloud.size());


    //Disable Voxel filter
    if(subsample_leaf_size > 0 && input_cloud.points.size() > 0)
    {
        pcl::VoxelGrid<PointRGBT> voxel_grid;
        voxel_grid.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> > (input_cloud));
        voxel_grid.setLeafSize (subsample_leaf_size, subsample_leaf_size, subsample_leaf_size);
        voxel_grid.filter (input_cloud);
    }    

    ROS_INFO("After Voxelgrid filter  %d points for plane segmentation",(int)input_cloud.size());
    
    //Disabled outlier filter
    //Filter outliers
    pcl::RadiusOutlierRemoval<PointRGBT> outlier_filter; //remove outlier
    outlier_filter.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> > (input_cloud));
    outlier_filter.setRadiusSearch (_outlier_radius);
    outlier_filter.setMinNeighborsInRadius(_outlier_nn);
    outlier_filter.filter(input_cloud);

    ROS_INFO("After outlier filter  %d points for plane segmentation",(int)input_cloud.size());
    ros::Time start=ros::Time::now();

    pcl::SACSegmentationFromNormals<PointRGBT, pcl::Normal> seg; // the ransac filter
    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (_ransac_iterations);
    seg.setDistanceThreshold (_ransac_dist_threshold);
    seg.setNormalDistanceWeight ( _ransac_normal_dist_weight);


    //for the Ransac using plane normals
    pcl::NormalEstimation<PointRGBT, pcl::Normal> ne;
    // Estimate point normals
    ne.setSearchMethod (boost::make_shared<pcl::search::KdTree<PointRGBT> > ());
    ne.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> >(input_cloud));
    ne.setKSearch (30);
    ne.compute (cloud_normals);  

    unsigned int retry=0; //counts number of retries if given Model doesn't contain enough points to be selected

    int objectNr=1;
    data_file_.open("plane_features.txt");

    while (retry < 10) {
    seg.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> >(input_cloud));
	seg.setInputNormals (boost::make_shared<pcl::PointCloud<pcl::Normal> >(cloud_normals));
	// Obtain the plane inliers and coefficients
	seg.segment (inliers_plane, coefficients_plane);
      
	if (inliers_plane.indices.size()!=0){

	    ROS_INFO("Found pieces");
        pcl::ProjectInliers<PointRGBT> projectedPlane;
	    //Project inliers onto plane model
	    projectedPlane.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        projectedPlane.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> >(input_cloud));
	    projectedPlane.setIndices (boost::make_shared<pcl::PointIndices> (inliers_plane)); 
        projectedPlane.setModelCoefficients (boost::make_shared< pcl::ModelCoefficients>(coefficients_plane));
	    projectedPlane.filter (cloud_plane);
	    
	    if (cloud_plane.points.size ()< RANSAC_THRESHOLD){
		retry++;
		ROS_INFO("Pieces to small");
		continue;
	    }
	    ROS_INFO("PointCloud representing the planar component: %d data points.", (int)cloud_plane.points.size ());

	    	    
	    //segment planes into different clusters
        //do clusterize for thesis
        /**********************************/
        pcl::EuclideanClusterExtraction< PointRGBT > clusterExtraction; //class for cluster extraction as the name says ;-)
	    std::vector< pcl::PointIndices > clusters; //container for indices of extracted plane segments
	    
        clusterExtraction.setInputCloud(boost::make_shared<pcl::PointCloud<PointRGBT> > (cloud_plane));
        clusterExtraction.setSearchMethod(boost::make_shared<pcl::search::KdTree<PointRGBT> > ());
	    clusterExtraction.setMinClusterSize (min_cluster_size_);
	    clusterExtraction.setClusterTolerance (_segmentation_cluster_dist);
        clusterExtraction.extract (clusters);

        ROS_INFO("Found %d clusters in plane.",(int) clusters.size());

	    for (size_t i=0;i<clusters.size();i++){	  		

            Plane plane;
            unsigned char r=rand() % 256;
            unsigned char g=rand() % 256;
            unsigned char b=rand() % 256;
            pcl::PointCloud<PointRGBT> plane_cluster;
            pcl::copyPointCloud (cloud_plane, clusters[i], plane_cluster);
            pcl::PointCloud<PointRGBT>::iterator iter;

            for (iter=plane_cluster.begin();iter!=plane_cluster.end();iter++){
                        iter->r=r;
                        iter->g=g;
                        iter->b=b;
                        plane.addPoint(Shared_Point(new Point((double)iter->x,(double)iter->y,(double)iter->z)));
                    }



            boost::shared_ptr<cv::Mat> image;
            double scale;

            /*DEBUG SHOW IMAGE*
            image=plane.getHullCVImages(scale);
            cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
            cv::imshow( "Display window", *image );
            cv::waitKey(0);
            ************************/

            //
            //recieves and stores alpha shapes (non filled) as images on disk
            image=plane.getHullCVImagesNofill(scale);


            std::stringstream shape_name;
            shape_name<<"./shapes/alpha_shape_"<<objectNr<<".png";
            try{
                cv::imwrite(shape_name.str().c_str(),*image);
            }
            catch (cv::Exception e)
            {
                std::cout<<e.what()<<std::endl;
            }



            //extract feautures from plane and print them
            Eigen::Vector3d pos,normal;
            RotatedRectangle max_extension;

            data_file_<<"Object Nr: "<<objectNr<<std::endl;
            data_file_<<"Size of Plane: "<<plane.getSizeOfPlane()<<std::endl;
            data_file_<<"Rectangularness: "<<plane.getRectangularness()<<std::endl;
            pos=plane.getPosition();
            data_file_<<"Position: "<<pos[0]<<","<<pos[1]<<","<<pos[2]<<std::endl;
            normal=plane.getNormalVector();
            data_file_<<"Normal: "<<normal[0]<<","<<normal[1]<<","<<normal[2]<<std::endl;
            max_extension=plane.getMaxExpansion();
            data_file_<<"Max Extension (w,h): "<< max_extension.width<<", "<<max_extension.height<<std::endl;
            data_file_<<"##############################"<<std::endl;
            objectNr++;


            pcl::toROSMsg(plane_cluster, cloud_plane_msg);
            _pub_plane.publish(cloud_plane_msg);
        }
        /*****************************************************/
        //publish segments without clusterization
        /*************************************
        unsigned char r=rand() % 256;
        unsigned char g=rand() % 256;
        unsigned char b=rand() % 256;

        pcl::PointCloud<PointRGBT>::iterator iter;
        //give the cloud some color
        for (iter=cloud_plane.begin();iter!=cloud_plane.end();iter++){
            iter->r=r;
            iter->g=g;
            iter->b=b;
        }
        pcl::toROSMsg(cloud_plane, cloud_plane_msg);
        _pub_plane.publish(cloud_plane_msg);
        **********************************/
		plane_bag_msg.plane_bag.push_back(cloud_plane_msg);


        pcl::ExtractIndices<PointRGBT> extract; //for point extraction from the filter
        pcl::ExtractIndices<pcl::Normal> extract_normals; //for normal extraction from the filter
        // Extract the planar inliers from the input cloud
        extract.setInputCloud (boost::make_shared<pcl::PointCloud<PointRGBT> > (input_cloud));
	    extract.setIndices (boost::make_shared<pcl::PointIndices> (inliers_plane)); 
	    
	    // Extract the NORMALS from the NORMAL cloud
	    extract_normals.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::Normal> > (cloud_normals));
	    extract_normals.setIndices (boost::make_shared<pcl::PointIndices> (inliers_plane));
	    
	    //extract the residuals from the point cloud
	    extract.setNegative (true);
        extract.filter(input_cloud);
	  
	    extract_normals.setNegative (true);
	    extract_normals.filter(cloud_normals);
	}
	else{ 
	    ROS_INFO("No model found");
	    break;
	}
    }        

    ros::Time stop=ros::Time::now();
    ros::Duration elapsed=stop-start;
    std::cout<<"Time elapsed for segmentation: " << elapsed.sec<<","<<elapsed.nsec<<std::endl;

    //Publish the bag of planes within one message
    _pub_plane_bag.publish(plane_bag_msg);
    //Publish residuals for further processing
    pcl::toROSMsg(input_cloud, residual_msg);
    _pub_residuals.publish(residual_msg);
    data_file_.close();

  }






  
private:

    ros::NodeHandle _n;
    ros::Publisher _pub_plane;  
    ros::Publisher _pub_plane_bag;
    ros::Publisher _pub_residuals;    
    ros::Subscriber _sub; 
    ros::ServiceServer  _srv_state;
        
    int _ransac_iterations;
    double _ransac_dist_threshold;
    double _ransac_normal_dist_weight;
    double _segmentation_cluster_dist;
    int min_cluster_size_;

    double _outlier_radius;
    int _outlier_nn;

    double subsample_leaf_size;
    int state;
    std::string _srv_state_name;
    std::ofstream data_file_;
};


int main (int argc, char** argv)
{
  ros::init (argc, argv, nodeName);
  ros::NodeHandle n;
  PlaneSegmentation plane(n);
  ros::spin ();
  return 0;
}
