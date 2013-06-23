/*
 * Markus Eich 2013
 * DFKI GmbH
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <impera_shape_msgs/PlaneBag.h>
#include <regionGrowing.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>
#include <plane.h>
#include <region_segmentation/set_params.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointIT;



std::string nodeName="region_segmentation";


class RegionSegmentation: RegionGrowing{

public:

  RegionSegmentation(ros::NodeHandle &n):n_(n){


    sub_=n_.subscribe("/cloud",1,&RegionSegmentation::scan_callback,this);
    pub_plane_=n_.advertise<sensor_msgs::PointCloud2> ("planes",10);
    pub_hull_=n_.advertise<sensor_msgs::PointCloud2> ("hulls",10);
    pub_plane_bag_=n_.advertise<impera_shape_msgs::PlaneBag> ("plane_bag",2);
    pub_residuals_=n_.advertise<sensor_msgs::PointCloud2> ("plane_residuals",10);
    set_param_srv_=n_.advertiseService("/set_params",&RegionSegmentation::setParams,this);

    //set sane values

    //Maximum distance between points and the plane.
    this->setGamma(0.08);
    //Minimum number of points in the plane.
    this->setTheta(800);
    //Maximum MSE for planes.
    this->setEpsilon(0.03);
    // Maximum distance between each point and its neighbors.
    this->setDelta(0.2);
    //set KNN
    this->setNeighbours(5);
    this->setBoostCallback(boost::bind(&RegionSegmentation::newPlaneReady,this,_1));

    srand (time(NULL));//random seed for random colors

    data_file_.open ("growing_data.txt",std::ios::app);
    data_file_<< "Params:"<<std::endl;
    data_file_<< "Dist Point Plane:"<<gamma<<std::endl;
    data_file_<< "Min Points in Plane:"<<theta<<std::endl;
    data_file_<< "MSE:"<<epsilon<<std::endl;
    data_file_<< "Delta:"<<delta<<std::endl;
    data_file_<< "KNN:"<<neighbours_count<<std::endl;
    data_file_<< "#####################################"<<std::endl;
    data_file_.close();

  }

  /*
   * Service call callback from ROS service. Sets paramters to the segmentation algorithm
   */
  bool setParams(region_segmentation::set_params::Request& req,region_segmentation::set_params::Request& resp){

      //Maximum distance between points and the plane.
      this->setGamma(req.point_plane_distance);
      //Minimum number of points in the plane.
      this->setTheta(req.min_points);
      //Maximum MSE for planes.
      this->setEpsilon(req.mse);
      // Maximum distance between each point and its neighbors.
      this->setDelta(req.point_neighbor_distance);
      //set KNN
      this->setNeighbours(req.KNN);

      return true;

  }
  /*
   * Callback funtion for segmentation. Called from segmenter single shape is ready
   */
  void newPlaneReady(boost::shared_ptr<Plane> cur_plane)
     {

      std::cout<<"Plane ready"<<std::endl;

      pcl::PointCloud<PointIT> pcl_cloud; //for easy conversion
      sensor_msgs::PointCloud2 cloud_msg;      
      boost::shared_ptr<cv::Mat> image;
      double scale;

      //get hull for drawing polygons as markers      
      std::list<std::list<std::pair<Shared_Point,Shared_Point> > > alpha_shape;
      alpha_shape=cur_plane->getHull();

      int colorr=rand() % 256;
      int colorg=rand() % 256;
      int colorb=rand() % 256;

      for( std::set<Shared_Point>::iterator xit = cur_plane->points.begin(); xit != cur_plane->points.end(); xit++) {                              
          PointIT point(colorr,colorg,colorb);
          point.x=(*xit)->p[0];
          point.y=(*xit)->p[1];
          point.z=(*xit)->p[2];
          pcl_cloud.push_back(point);
      }

      pcl::toROSMsg(pcl_cloud, cloud_msg);

      cloud_msg.header.frame_id="/base_link";
      cloud_msg.header.stamp=ros::Time::now();

      pub_plane_.publish(cloud_msg);

      /*debug block. Gets and shows filled alpha hull
      image=cur_plane->getHullCVImages(scale);
      cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
      cv::imshow( "Display window", *image );
      cv::waitKey(0);
      /*end debug block*/

     }
/*
 * Callback funtion for the ROS subscriber. Receives a point cloud message
 */
  void scan_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_ptr){
 

	  size_t n_p;
	  size_t p_i=0;
      int objectNr=1;

      ROS_INFO("received point cloud");

      pcl::PointCloud<PointT> pcl_cloud; //for easy conversion
      pcl::PointCloud<PointT> pcl_cloud_pt_filtered; //reduce point density
      //pcl::RadiusOutlierRemoval<PointT> outlier_filter; //remove outlier

       pcl::fromROSMsg(*cloud_ptr, pcl_cloud);

      if(pcl_cloud.points.size() > 0)
         {
             pcl::VoxelGrid<PointT> sor;
             sor.setInputCloud (boost::make_shared<pcl::PointCloud<PointT> > (pcl_cloud));
             sor.setLeafSize (0.02, 0.02, 0.02);
             sor.filter (pcl_cloud_pt_filtered);
         }


          ROS_INFO("Received %d points for plane segmentation",(int)pcl_cloud_pt_filtered.size());
          n_p = pcl_cloud_pt_filtered.size();
          std::vector<Shared_Point> points(n_p);

          pcl::PointCloud<PointT>::iterator pIter;//point iterator

          for (pIter=pcl_cloud_pt_filtered.begin(); pIter!=pcl_cloud_pt_filtered.end(); pIter++){

                  points[p_i++] = Shared_Point(new Point((double)pIter->x,(double)pIter->y,(double)pIter->z));
          }


        this->setPointcloud(points);



      result_.clear();
      ROS_INFO("Start the growing process. Lets see..");
      ros::Time start=ros::Time::now();
      this->regionFind(result_);
      ros::Time stop=ros::Time::now();
      ros::Duration elapsed=stop-start;

      std::cout<<"Time elapsed: " << elapsed.sec<<","<<elapsed.nsec<<std::endl;

      ROS_INFO("Found %d regions",(int)result_.size());

      std::list<Shared_Plane>::iterator  iter;

      data_file_.open("growing_data.txt",std::ios::app);

      data_file_<<"Number of Planes: "<<(int)result_.size()<<std::endl;
      data_file_<<"##############################"<<std::endl;
      data_file_<<"##############################"<<std::endl;
      Shared_Plane plane;

      double scale;
      double ortho;
      double paral;

      for (iter=result_.begin();iter!=result_.end();iter++){
            boost::shared_ptr<cv::Mat> image;
            plane=*iter;
            Eigen::Vector3d pos;
            Eigen::Vector3d normal;
            Eigen::Vector3d inertiaX;
            Eigen::Vector3d inertiaY;
            Eigen::Vector3d ground_normal(0,0,1);
            RotatedRectangle max_extension;
          data_file_<<"Object Nr: "<<objectNr<<std::endl;
          data_file_<<"Size of Plane: "<<plane->getSizeOfPlane()<<std::endl;
          data_file_<<"Rectangularness: "<<plane->getRectangularness()<<std::endl;
          pos=plane->getPosition();
          data_file_<<"Position: "<<pos[0]<<","<<pos[1]<<","<<pos[2]<<std::endl;
          inertiaX=plane->getInertiaAxisX();
          data_file_<<"Intertia AxisX: "<<inertiaX[0]<<","<<inertiaX[1]<<","<<inertiaX[2]<<std::endl;
          inertiaY=plane->getInertiaAxisY();
          data_file_<<"Intertia AxisY: "<<inertiaY[0]<<","<<inertiaY[1]<<","<<inertiaY[2]<<std::endl;
          normal=plane->getNormalVector();
          data_file_<<"Normal: "<<normal[0]<<","<<normal[1]<<","<<normal[2]<<std::endl;
          max_extension=plane->getMaxExpansion();
          ortho=fabs(1-fabs(ground_normal.dot(normal)));
          paral=fabs(1-fabs((ground_normal.dot(normal)-(ground_normal.norm()*normal.norm()))));
          data_file_<<"orthoG: "<<ortho<<std::endl;
          data_file_<<"paralG: "<<paral<<std::endl;
          data_file_<<"Max Extension (w,h): "<< max_extension.width<<", "<<max_extension.height<<std::endl;
          data_file_<<"##############################"<<std::endl;

          //recieves and stores alpha shapes (non filled) as images on disk
          image=plane->getHullCVImagesNofill(scale);

          /*Debug Block for showing shapes which have been extracted
          cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
          cv::imshow( "Display window", *image );
          cv::waitKey(0);
          //End debug block*/

          std::stringstream shape_name;
          shape_name<<"./shapes/alpha_shape_"<<objectNr<<".png";
          try{
            cv::imwrite(shape_name.str().c_str(),*image);
          }
          catch (cv::Exception e)
          {
              std::cout<<e.what()<<std::endl;
          }


          objectNr++;
      }

      data_file_.close();

  }
  


private:

    std::list<Shared_Plane> result_;

    std::ofstream data_file_;
    ros::NodeHandle n_;

    ros::Publisher pub_plane_;  
    ros::Publisher pub_hull_;
    ros::Publisher pub_plane_bag_;
    ros::Publisher pub_residuals_;    
    ros::Subscriber sub_; 
    ros::ServiceServer set_param_srv_;    
 };


int main (int argc, char** argv)
{
  ros::init (argc, argv, nodeName);
  ros::NodeHandle n;
  RegionSegmentation plane(n);
  ros::spin ();
  return 0;
}
