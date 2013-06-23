/****
  * Markus Eich
  * 2013
  * DFKI GmbH
  *
  */


#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/MarkerArray.h"

#include <pcl/ModelCoefficients.h>

#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/normal_3d.h>
#include <pcl/common/angles.h>

#include <tf/tf.h>

#include <impera_shape_msgs/CylinderShapeBag.h>
//#include <reasoner_msgs/GetCylinders.h>


#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <FuzzyDLProducer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cstdlib>
#include <boost/thread.hpp>
#include <boost/foreach.hpp>


typedef pcl::PointXYZ PointT;



class CylinderSegmentation{

public:

  CylinderSegmentation(ros::NodeHandle &n):n_(n){

    ros::param::param("~ransac_iterations",(int&) ransac_iterations_, 10000);
    ros::param::param("~ransac_dist_thresold",(double&) ransac_dist_threshold_, 0.05);//0.05
    ros::param::param("~ransac_normal_dist_weight",(double&) ransac_normal_dist_weight_, 0.05);//0.04
    ros::param::param("~ransac_min_radius",(double&) ransac_min_radius_, 0.01);
    ros::param::param("~ransac_max_radius",(double&) ransac_max_radius_, 0.50);
    ros::param::param("~min_likelihood",(double&) min_likelihood_,0.10);
    ros::param::param("~input_topic",(std::string&) input_topic_, std::string("/plane_residuals"));


    sub_=n_.subscribe(input_topic_.c_str(),1,&CylinderSegmentation::scan_callback,this);
    pub_cylinder_=n_.advertise<sensor_msgs::PointCloud2> ("/cylinders",10);
    pub_cylinder_bag_=n_.advertise<impera_shape_msgs::CylinderShapeBag> ("/cylinder_features_bag",2);
    pub_residuals_=n_.advertise<sensor_msgs::PointCloud2> ("/cylinder_residuals",10);
    pub_cluster_=n_.advertise<sensor_msgs::PointCloud2> ("/clusters",100);
    pub_cylinder_marker_=n_.advertise<visualization_msgs::MarkerArray>("cylinder_marker",10);

    //reasoner_client_=_n.serviceClient<reasoner_msgs::GetCylinders>("/doReasoning");    

  }


  /**
   *Segments the input point cloud into several clusters
   * @param input_cloud_ptr
   * @param clusters
   */
	void segment_clusters(const pcl::PointCloud<PointT>::Ptr input_cloud_ptr,
			std::vector<pcl::PointCloud<PointT> > & clusters) {

		pcl::search::KdTree<PointT>::Ptr cluster_tree (new pcl::search::KdTree<PointT> ());

		std::vector<pcl::PointIndices> cluster_indices;
		sensor_msgs::PointCloud2 cluster_msg;

		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(10000);
		ec.setSearchMethod(cluster_tree);
		ec.setInputCloud(input_cloud_ptr);
		ec.extract(cluster_indices);


        ROS_INFO("Found %d Clusters\n",(int)cluster_indices.size());

		for (std::vector<pcl::PointIndices>::const_iterator it =cluster_indices.begin(); it != cluster_indices.end(); ++it) {
			pcl::PointCloud<PointT>::Ptr cloud_cluster( new pcl::PointCloud<PointT>);
			for (std::vector<int>::const_iterator pit = it->indices.begin();pit != it->indices.end(); pit++)
				cloud_cluster->points.push_back(input_cloud_ptr->points[*pit]); //*
			cloud_cluster->width = cloud_cluster->points.size();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;

            ROS_INFO("Cluster Size: %d \n",(int)cloud_cluster->points.size());
			//adds a cluster to the output list
			clusters.push_back(*cloud_cluster);

			pcl::toROSMsg(*cloud_cluster, cluster_msg);

			cluster_msg.header.frame_id = frame_id_;
			cluster_msg.header.stamp=ros::Time::now();

			pub_cluster_.publish(cluster_msg);


            /*slow down ouput
            ros::Rate r(2);
			r.sleep();
            */




		}

	}
	/**
	 *
	 * @param input_cloud_ptr The input point cloud
	 * @param max_pt the maximum point
	 * @param min_pt the minimum point
	 * @param coefficients the coefficents coming from the segmentation
	 * @return the height of the cylinder
	 */

    double get_cylinder_height(const pcl::PointCloud<PointT>::Ptr input_cloud_ptr, PointT* max_pt, PointT* min_pt, const pcl::ModelCoefficients coefficients){

        Eigen::Vector3f orientation(coefficients.values[3],coefficients.values[4],coefficients.values[5]);

        Eigen::Vector3f pointOnAxis(coefficients.values[0],coefficients.values[1],coefficients.values[2]);
        Eigen::Vector3f max_point;
        Eigen::Vector3f min_point;
        Eigen::Vector3f pointTranslated;

        pointTranslated=input_cloud_ptr->points[0].getVector3fMap()-pointOnAxis;

		double min=(pointTranslated.dot(orientation))/orientation.norm();
		double max=(pointTranslated.dot(orientation))/orientation.norm();
		min_point=pointOnAxis+min*orientation;
		max_point=pointOnAxis+max*orientation;

		BOOST_FOREACH(const PointT& pt, input_cloud_ptr->points){

			//Translate all points in the direction of the axis point
			pointTranslated=pt.getVector3fMap()-pointOnAxis;

			//Project all points on the unit vector and comparing the scalar product
			if (pointTranslated.dot(orientation)<min){
				min=(pointTranslated.dot(orientation))/orientation.norm();
				//retransform the point to the original axis
				min_point=pointOnAxis+min*orientation;
			}
			if (pointTranslated.dot(orientation)>max){
				max=(pointTranslated.dot(orientation))/orientation.norm();
				//retransform the point to the original axis
				max_point=pointOnAxis+max*orientation;
			}
		}

		//Conversion back to Point Type
		max_pt->x=max_point[0];
		max_pt->y=max_point[1];
		max_pt->z=max_point[2];

		min_pt->x=min_point[0];
		min_pt->y=min_point[1];
		min_pt->z=min_point[2];

		return max-min;


	}


	/**
	 * Gets the similarity between a cylinder and a given point cloud between 0.0 and 1.0
	 * @param input_cloud_ptr the input cloud
	 * @return likelihood
	 */
	double get_cylinder_likelihood(
            const pcl::PointCloud<PointT>::Ptr input_cloud_ptr, pcl::ModelCoefficients& coefficients) {

		//The KD tree for the segmentation
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
		//Structure for normal estimation
		pcl::NormalEstimation<PointT, pcl::Normal> ne; //Normal estimation

		//for the Ransac using Cylinder normals
		pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; // the ransac filter

		//the structure to store the cloud normals
		pcl::PointCloud<pcl::Normal> cloud_normals; // the cloud normals

		//for extraction
		pcl::ExtractIndices<PointT> extract; //for point extraction from the filter
		pcl::ExtractIndices<pcl::Normal> extract_normals;

		//all points within a cylinder
		pcl::PointCloud<PointT> cloud_out;

		//for publishing
		sensor_msgs::PointCloud2 cloud_out_msg;
		sensor_msgs::PointCloud2 residual_msg;
        //impera_shape_msgs::CylinderShapeBag cylinder_bag_msg;

		//the sphere coefficents generated by segmentation
		pcl::PointIndices inliers;


      // Estimate point normals
		ne.setSearchMethod(tree);
		ne.setInputCloud(input_cloud_ptr);
		ne.setKSearch(50);
		ne.compute(cloud_normals);

		// Create the segmentation object for the planar model and set all the parameters
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_CYLINDER);
		seg.setMethodType(pcl::SAC_RANSAC);
        seg.setNormalDistanceWeight(ransac_normal_dist_weight_);
		seg.setMaxIterations(ransac_iterations_);
		seg.setDistanceThreshold(ransac_dist_threshold_);
		seg.setRadiusLimits(ransac_min_radius_, ransac_max_radius_);

		seg.setInputCloud(input_cloud_ptr);
		seg.setInputNormals(boost::make_shared<pcl::PointCloud<pcl::Normal> >(cloud_normals));
		// Obtain the cylinder inliers and coefficients
		seg.segment(inliers, coefficients);

        ROS_INFO("Found %d inliers in a cloud of %d points",(int) inliers.indices.size(), (int) input_cloud_ptr->size());

		// Extract the inliers from the input cloud

		if (inliers.indices.size() > 0) {
			extract.setInputCloud(input_cloud_ptr);
			extract.setIndices(boost::make_shared<const pcl::PointIndices>(inliers));

			extract_normals.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::Normal> >(cloud_normals));
			extract_normals.setIndices(boost::make_shared<const pcl::PointIndices>(inliers));

			extract.setNegative(false);
			extract.filter(cloud_out);	

			//publish segments
			pcl::toROSMsg(cloud_out, cloud_out_msg);

			cloud_out_msg.header.frame_id = frame_id_;
			pub_cylinder_.publish(cloud_out_msg);


			//Publish residuals for further processing
			pcl::toROSMsg(*input_cloud_ptr, residual_msg);
			residual_msg.header.frame_id = frame_id_;
			pub_residuals_.publish(residual_msg);


			return static_cast<double>(cloud_out.points.size())/input_cloud_ptr->size();

		} else {
			return 0.0;
		}
}



  void scan_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg){

      //local variables
      double likelihood=0.0;
      double height=0.0;
      int id=1;
      Fuzzy::FuzzyInstance instance;
      std::vector<pcl::PointCloud<PointT> >::iterator cluster_iter;
      visualization_msgs::MarkerArray marker_array;
      impera_shape_msgs::CylinderShapeBag cylinder_shape_bag;
      pcl::ModelCoefficients coefficients;
      pcl::PointCloud<PointT> input_cloud;// the raw input point
      std::vector<pcl::PointCloud<PointT> > clusters;
      Fuzzy::FuzzyDLProducer producer;
      ///////////////////////////////////////////////

      //take the same frame id as the original cloud;
      if (cloud_msg->header.frame_id.empty())
    	  frame_id_="/base_link";
      else
    	  frame_id_=cloud_msg->header.frame_id;

      pcl::fromROSMsg(*cloud_msg, input_cloud);
      ROS_INFO("Cylinder segmentation received %d points",(int)input_cloud.size());

      //clusterize point cloud before applying Ransac
      this->segment_clusters(boost::make_shared<pcl::PointCloud<PointT> >(input_cloud), clusters);

      data_file_.open("cluster_features.txt");

      for (cluster_iter=clusters.begin();cluster_iter!=clusters.end();cluster_iter++){

    	  impera_shape_msgs::CylinderShape cylinder_shape;
    	  PointT maxPointOnAxis;
    	  PointT minPointOnAxis;

          likelihood=get_cylinder_likelihood(boost::make_shared<pcl::PointCloud<PointT> >(*cluster_iter),coefficients);

          if (coefficients.values.size()!=7){ //check if model is found otherwise continue;
              continue;
          }

          //Debug coefficients
          //std::cout << "Coefficients: "<<coefficients<<std::endl;
          PointT pointOnAxis(coefficients.values[0],coefficients.values[1],coefficients.values[2]);
          tf::Vector3 axis_vector(coefficients.values[3],coefficients.values[4],coefficients.values[5]);

          height=get_cylinder_height(boost::make_shared<pcl::PointCloud<PointT> >(*cluster_iter),&maxPointOnAxis, &minPointOnAxis, coefficients);

          //calculate the Orientation of the Cylinder in Quaternions
          tf::Vector3 up_vector(0.0, 0.0, 1.0);
          tf::Vector3 right_vector = axis_vector.cross(up_vector);
          right_vector.normalize();
          tf::Quaternion q(right_vector, -1.0*acos(axis_vector.dot(up_vector)));
          q.normalize();
          geometry_msgs::Quaternion cylinder_orientation;
          geometry_msgs::Point cylinder_position;
          geometry_msgs::Pose label_pose;
      	  label_pose.orientation.w = 1.0;

          tf::quaternionTFToMsg(q, cylinder_orientation);

          cylinder_position.x=(maxPointOnAxis.x+minPointOnAxis.x)/2.0;
          cylinder_position.y=(maxPointOnAxis.y+minPointOnAxis.y)/2.0;
          cylinder_position.z=(maxPointOnAxis.z+minPointOnAxis.z)/2.0;

          label_pose.position.x=maxPointOnAxis.x;
          label_pose.position.y=maxPointOnAxis.y;
          label_pose.position.z=maxPointOnAxis.z+0.30;


    	  if (likelihood>=min_likelihood_){


    		  visualization_msgs::Marker marker;
    		  visualization_msgs::Marker label;

    		  marker.ns = std::string("cylinders");
    		  marker.header.frame_id = frame_id_ ;
    		  marker.id = id;
              marker.type = visualization_msgs::Marker::CYLINDER;
    		  marker.action = visualization_msgs::Marker::ADD;
              marker.scale.x = coefficients.values[6]*2.0;
              marker.scale.y = coefficients.values[6]*2.0;
              marker.scale.z = height;
              marker.color.g = 0.5;
              marker.color.r = 0.5;
              marker.color.b = 0.0;
    		  marker.color.a = 0.8;
              marker.pose.orientation=cylinder_orientation;
              marker.pose.position=cylinder_position;
              marker.lifetime = ros::Duration(10);
    		  marker.header.stamp = ros::Time::now();

    		  label.ns = std::string("labels");
    		  label.header.frame_id = frame_id_ ;
    		  label.id = id;
    		  label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    		  label.action = visualization_msgs::Marker::ADD;
    		  label.text=std::string("Object "+boost::lexical_cast<std::string>(id));
    		  label.scale.x = 0.2;
    		  label.scale.y = 0.2;
    		  label.scale.z = 0.2;
    		  label.color.g = 1.0;
    		  label.color.r = 1.0;
    		  label.color.b = 0.0;
    		  label.color.a = 1.0;
    		  label.pose=label_pose;
    		  label.lifetime = ros::Duration(10);
    		  label.header.stamp = ros::Time::now();


              marker_array.markers.push_back(marker);
              marker_array.markers.push_back(label);

              //TODO send correct features as a shape

              cylinder_shape.height=height;

              //Take the base of the object as the position for the reasoner. It is important
              //where the object is standing
              cylinder_shape.position.x=cylinder_position.x;
              cylinder_shape.position.y=cylinder_position.y;
              cylinder_shape.position.z=cylinder_position.z;

              cylinder_shape.orientation.x=1;
              cylinder_shape.orientation.y=0;
              cylinder_shape.orientation.z=0;

              cylinder_shape.likelihood=likelihood;
              cylinder_shape.radius= coefficients.values[6];


              cylinder_shape_bag.cylinder_shape_bag.push_back(cylinder_shape);       
    	
    		  instance.name=std::string("cylinder"+boost::lexical_cast<std::string>(id));
    		  instance.hasCylinderType=likelihood;
              instance.hasCylinderHeight=cylinder_shape.height;
              instance.hasCylinderRadius=cylinder_shape.radius;
              instance.hasX=cylinder_shape.position.x;
              instance.hasY=cylinder_shape.position.y;
              instance.hasZ=cylinder_shape.position.z;
              instance.isOthogonal=(1.0-std::max(fabs(tf::Vector3(1.0,0.0,0).dot(axis_vector)),fabs(tf::Vector3(0.0,1.0,0).dot(axis_vector))));
              instance.isParallel=std::max(fabs(tf::Vector3(1.0,0.0,0).dot(axis_vector)),fabs(tf::Vector3(0.0,1.0,0).dot(axis_vector)));
              //instance.isOthogonal=1.0-tf::Vector3(1.0,1.0,0).dot(axis_vector);
              //instance.isParallel=tf::Vector3(1.0,1.0,0).dot(axis_vector);
    		  producer.AddInstance(instance);


             //write extracted features



             data_file_<<"Object Nr: "<<id<<std::endl;
             data_file_<<"Height: "<<cylinder_shape.height<<std::endl;
             data_file_<<"Radius: "<<cylinder_shape.radius<<std::endl;
             data_file_<<"Position: "<<cylinder_shape.position.x<<","<<cylinder_shape.position.y<<","<<cylinder_shape.position.z<<std::endl;
             data_file_<<"Orientation: "<<axis_vector.getX()<<","<<axis_vector.getY()<<","<<axis_vector.getZ()<<std::endl;
             data_file_<<"##############################"<<std::endl;

             id++;
    		 pub_cylinder_marker_.publish(marker_array);

    	  }

    	  pub_cylinder_marker_.publish(marker_array);
            //_pub_cylinder_bag.publish(cylinder_shape_bag);          

      }
      data_file_.close();

      //Save the instances to a file for later reasoning
      //file has to be saved under the same name in any case

      if (std::getenv("ROS_HOME")==NULL){
    	  producer.GenerateKB(std::string(std::getenv("HOME"))+std::string("/.ros/")+std::string(Fuzzy::CYLINDER_KB_FILENAME));
      }
      else
    	  producer.GenerateKB(std::string(std::getenv("ROS_HOME"))+std::string("/")+std::string(Fuzzy::CYLINDER_KB_FILENAME));


      /*
      //Message call to get the likelihood of the cylinders
      reasoner_msgs::GetCylinders getCylinders;
      reasoner_client_.call(getCylinders);

      std::vector<reasoner_msgs::Reasoner>::iterator iter=getCylinders.response.result.begin();

      while (iter!=getCylinders.response.result.end()){

    	  std::cout << (*iter).name<<": "<<(*iter).likelihood <<std::endl;
    	  iter++;
      }

      */


  }

  
private:

    ros::NodeHandle n_;
    ros::Publisher pub_cylinder_;
    ros::Publisher pub_cylinder_bag_;
    ros::Publisher pub_residuals_;
    ros::Publisher pub_cluster_;
    ros::Publisher pub_cylinder_marker_;
    ros::ServiceClient reasoner_client_;
    ros::Subscriber sub_;
    
    int ransac_iterations_;
    double ransac_dist_threshold_;
    double ransac_normal_dist_weight_;
    double ransac_min_radius_;
    double ransac_max_radius_;
    double min_likelihood_;
    std::string frame_id_;
    std::string input_topic_;
    std::ofstream data_file_;

};


int main (int argc, char** argv)
{
  ros::init (argc, argv, "cylinder_segmentation");
  ros::NodeHandle n;
  /*ROS_INFO("Waiting for /callme");
  ros::service::waitForService("/callme");
  ROS_INFO("Found /callme");
  */

  CylinderSegmentation cylinder(n);

  ros::spin();

  ROS_INFO("Exit cylinder segmentation");
  return 0;
}



