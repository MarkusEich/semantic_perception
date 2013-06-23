/*
 * Author
 * Markus Eich 2011
 *
 * markus.eich@dfki.de
 * Class to extract plane features from segmented 3D point cloud planes
*/
#include <assert.h>

#include <ros/ros.h>

#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <shape_msgs/Shape.h>
#include <shape_msgs/ShapeBag.h>
#include <shape_msgs/PlaneBag.h>


#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include "pcl/surface/convex_hull.h"
#include <pcl/surface/concave_hull.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/cartesian2d.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/assign.hpp>
#include <boost/geometry/algorithms/append.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//typedefs for PCL
typedef pcl::PointXYZ PointT;


class FeatureExtractor{

public:

    FeatureExtractor(ros::NodeHandle &n):_n(n){

	ros::param::param("~alpha_value",(double&) _alpha_value, 0.1);

	_sub=_n.subscribe("/plane_bag",2,&FeatureExtractor::scan_callback,this);
	_pub_hull=_n.advertise<sensor_msgs::PointCloud2> ("/plane_hull",1);
	_pub_bounding=_n.advertise<sensor_msgs::PointCloud2> ("/bounding_rect",1);
	_pub_alpha=_n.advertise<sensor_msgs::PointCloud2> ("/alpha_shape",1);
	_pub_shape_features=_n.advertise<shape_msgs::Shape> ("/shape_feature",1);
	_pub_shape_feature_bag=_n.advertise<shape_msgs::ShapeBag> ("/shape_feature_bag",1);
	_pub_alpha_polygon=_n.advertise<geometry_msgs::PolygonStamped> ("/alpha_shape_polygon",1);
	_pub_normal_marker_array=_n.advertise<visualization_msgs::MarkerArray> ("/normals_marker_array",10);
        //_pub_normal_marker=_n.advertise<visualization_msgs::Marker> ("/normals_marker",10);
	
  }

    /**
      Main callback loop when messages arives
      @param cloud_ptr the incomming PC message
     */

  void scan_callback(const shape_msgs::PlaneBag &plane_bag_msg){

    //for publishing
    shape_msgs::ShapeBag shape_feature_bag_msg;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> cogs;

    ROS_INFO("Received %d planes in a bag.", (int) plane_bag_msg.plane_bag.size());
    
    //TODO get planes out of the bag and process them
    for (size_t i=0;i< plane_bag_msg.plane_bag.size();i++){

	pcl::PointCloud<PointT> plane;// the received plane
	pcl::ModelCoefficients coeff_plane; //the plane coefficents  

	
	//all points within a hull
	pcl::PointCloud<PointT> convexHull;
	pcl::PointCloud<PointT> boundingRect;
	pcl::PointCloud<PointT> alphaShape;
	

	//structure to generate convex and concave hull
	pcl::ConvexHull<PointT> convex_hull;
	pcl::ConcaveHull<PointT> concave_hull;		

	sensor_msgs::PointCloud2 bounding_rect_msg;
	sensor_msgs::PointCloud2 alpha_shape_msg;
	sensor_msgs::PointCloud2 convex_hull_msg;  
	shape_msgs::Shape shape_feature_msg;
	geometry_msgs::PoseStamped pose_msg;     
	geometry_msgs::PolygonStamped polygon_msg;

	Eigen::Vector3f cog;

	pcl::fromROSMsg(plane_bag_msg.plane_bag[i], plane); 
	
	boundingRect.header=plane.header;//copy header from received plane
	alphaShape.header=plane.header;//copy header from received plane
	pose_msg.header=plane.header;//copy header from received plane
	polygon_msg.header=plane.header;//copy header from received plane
	_frame_id=plane.header.frame_id; //get right Frame ID
	
	ROS_DEBUG ("Received Plane with %d data points for feature extraction.", (int)plane.points.size ());	

	//get the plane coefficients for the received plane
	this->estimateCoefficients(plane.makeShared(), coeff_plane);

	convex_hull.setInputCloud (plane.makeShared());
	concave_hull.setInputCloud (plane.makeShared());    
	concave_hull.setAlpha(_alpha_value);
	

	//convex hull extraction
	convex_hull.reconstruct (convexHull);
	assert(convexHull.size()>0);
	ROS_DEBUG ("Convex Hull has %d data points.", (int)convexHull.size ());

	
	//calculate the alpha shapes
	concave_hull.reconstruct (alphaShape);
	ROS_DEBUG ("Alpha hull has %d data points.", (int)alphaShape.size ());
	assert(alphaShape.size()>0);
	
	//calculate the minimal bounding box
	this->calcBoundingBox(convexHull.makeShared(), boost::make_shared<pcl::ModelCoefficients>(coeff_plane), boundingRect);
	       

	//fill and publish messages
	pcl::toROSMsg (boundingRect,bounding_rect_msg);
	pcl::toROSMsg (alphaShape,alpha_shape_msg);
	pcl::toROSMsg (convexHull, convex_hull_msg);
	
	_pub_alpha.publish(alpha_shape_msg);
	_pub_bounding.publish(bounding_rect_msg);    
	_pub_hull.publish(convex_hull_msg);
	
	//set the points of the alpha shape in the shape feature
	for (size_t i=0;i<boundingRect.points.size();i++){
	    geometry_msgs::Point pt;
	    pt.x=boundingRect.points[i].x;
	    pt.y=boundingRect.points[i].y;
	    pt.z=boundingRect.points[i].z;
	    shape_feature_msg.bounding_shape.push_back(pt);
	}
	
	//set extracted features
	shape_feature_msg.rectangularness=this->calcRectangularness(alphaShape.makeShared(), boundingRect.makeShared(), boost::make_shared<pcl::ModelCoefficients>(coeff_plane));	
	shape_feature_msg.area=this->calcArea(alphaShape.makeShared(), boost::make_shared<pcl::ModelCoefficients>(coeff_plane));
	shape_feature_msg.extension=this->calcExtension(boundingRect.makeShared());       


	cog=this->calcCoG(alphaShape.makeShared(), boost::make_shared<pcl::ModelCoefficients>(coeff_plane));
	
	shape_feature_msg.position.x=cog[0];
	shape_feature_msg.position.y=cog[1];
	shape_feature_msg.position.z=cog[2];

	//correct the normal estimation consistent towards view port (0,0,0)
	float nx=coeff_plane.values[0];
	float ny=coeff_plane.values[1];
	float nz=coeff_plane.values[2];
	PointT point(cog[0],cog[1],cog[2]);

	pcl::flipNormalTowardsViewpoint(point,0,0,0,nx,ny,nz);

	shape_feature_msg.normal.x=nx;
	shape_feature_msg.normal.y=ny;
	shape_feature_msg.normal.z=nz;

	//publish the feature set
	_pub_shape_features.publish(shape_feature_msg);   
	shape_feature_bag_msg.shape_bag.push_back(shape_feature_msg);
	
	//publish sorted alpha shape as polygon       	
	for (size_t i=0;i<alphaShape.size();i++){
	    geometry_msgs::Point32 pt;
	    pt.x=alphaShape.points[i].x;
	    pt.y=alphaShape.points[i].y;
	    pt.z=alphaShape.points[i].z;
	    polygon_msg.polygon.points.push_back(pt);
	}
	_pub_alpha_polygon.publish(polygon_msg);   
	
	//store individual normals and cog for marker visualization
	normals.push_back(Eigen::Vector3f(nx,ny,nz));
	cogs.push_back(Eigen::Vector3f(cog[0],cog[1],cog[2]));

    }

    //publish the generated features in a bag in order to publish them together
    _pub_shape_feature_bag.publish(shape_feature_bag_msg);
    
    //pusblish the normals als a marker array
    this->showNormals(normals,cogs);


  }
    /**
       Function to estimate plane normal (coefficient)
      @param alpha_ptr input cloud
      @param coeff_plane output plane normal
     */
    void estimateCoefficients(pcl::PointCloud<PointT>::ConstPtr cloud_ptr,  pcl::ModelCoefficients& coeff_plane)
    {
	
	//use segmentation to estimate the plane coefficients
	pcl::SACSegmentation< PointT > seg;
	pcl::PointIndices inliers_plane;//the indices of points which lay in the plane
	seg.setInputCloud (cloud_ptr);
	seg.setOptimizeCoefficients (true);    
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setDistanceThreshold (0.01);
	seg.segment(inliers_plane,coeff_plane);

    }         
    /** Calculates the envelop of a given point cloud, i.e. the minimum bounding box
	@param cloud_ptr
	@param coeff_plane
	@param boundingRect
     */    	
    void calcBoundingBox(pcl::PointCloud<PointT>::ConstPtr cloud_ptr, pcl::ModelCoefficientsConstPtr coeff_plane ,pcl::PointCloud<PointT>& boundingRect){
	
	
	Eigen::Vector3f plane_normal,v,u;
	std::vector<cv::Point2f> points;

	plane_normal= Eigen::Vector3f(coeff_plane->values[0],
				      coeff_plane->values[1],
				      coeff_plane->values[2]);
	
	// compute an orthogonal normal to the plane normal	
	v = plane_normal.unitOrthogonal();
	
	// take the cross product of the two normals to get
	// a thirds normal, on the plane
	u = plane_normal.cross(v);
	
	// project the 3D point onto a 2D plane
	
	// choose a point on the plane
	Eigen::Vector3f p0(cloud_ptr->points[0].x,
			   cloud_ptr->points[0].y,
			   cloud_ptr->points[0].z);
    
	for(size_t i=0; i<cloud_ptr->points.size(); i++){	    
	    Eigen::Vector3f p3d(cloud_ptr->points[i].x,
				cloud_ptr->points[i].y,
				cloud_ptr->points[i].z);
	    
	    // subtract all 3D points with a point in the plane
	    // this will move the origin of the 3D coordinate system
	    // onto the plane
	    p3d = p3d - p0;

	    cv::Point2f p2d;
	    p2d.x = p3d.dot(u);
	    p2d.y = p3d.dot(v);
	    points.push_back(p2d);
	}
	
	cv::Mat points_mat(points);
	cv::RotatedRect rrect = cv::minAreaRect(points_mat);
	cv::Point2f rrPts[4];
	rrect.points(rrPts);	
	
	//reproject the points into 3D and store them as a pointcloud
	for(size_t i=0; i<4; i++)
	{       	
	    Eigen::Vector3f pbbx;
	    pbbx=rrPts[i].x*u;
	    pbbx+=rrPts[i].y*v;
	    pbbx+=p0;	
	    boundingRect.push_back(pcl::PointXYZ(pbbx[0],pbbx[1],pbbx[2]));
	}	       	
    }

    /** Calulates the rectangularness as a fuzzy value.
	@param alpha_ptr input cloud 1
	@param bounding_ptr input cloud 2
	@return fuzzy value how rectangular is the shape
     */
    double calcRectangularness(pcl::PointCloud<PointT>::ConstPtr alpha_ptr, pcl::PointCloud<PointT>::ConstPtr bounding_ptr, pcl::ModelCoefficientsConstPtr coeff_plane)
    {

	double area_alpha,area_envelop;

	area_alpha=this->calcArea(alpha_ptr,coeff_plane);
	area_envelop=this->calcArea(bounding_ptr,coeff_plane);

	return area_alpha/area_envelop;
    }
    
    /** Calulates the area of a given polygon. Polygon has to be ordered
	@param cloud_ptr input cloud
	@param coeff_plane normal of the plane needed for 3D-> 2D projection
	@return enclosed area     
     */

    double calcArea(pcl::PointCloud<PointT>::ConstPtr cloud_ptr, pcl::ModelCoefficientsConstPtr coeff_plane)
    {
	boost::geometry::polygon_2d polygon; 
	Eigen::Vector3f plane_normal,v,u;
	
	if  (cloud_ptr->size()==0) return 0.0f;

	plane_normal= Eigen::Vector3f(coeff_plane->values[0],
				      coeff_plane->values[1],
				      coeff_plane->values[2]);
	
	// compute an orthogonal normal to the plane normal	
	v = plane_normal.unitOrthogonal();
	
	// take the cross product of the two normals to get
	// a thirds normal, on the plane
	u = plane_normal.cross(v);
	
	// project the 3D point onto a 2D plane
	
	// choose a point on the plane
	Eigen::Vector3f p0(cloud_ptr->points[0].x,
			cloud_ptr->points[0].y,
			cloud_ptr->points[0].z);
    
	//project each point on a 2D plane for area estimation
	for(size_t i=0; i<cloud_ptr->points.size(); i++)
		{
			Eigen::Vector3f p3d(cloud_ptr->points[i].x,
					    cloud_ptr->points[i].y,
					    cloud_ptr->points[i].z);

			// subtract all 3D points with a point in the plane
			// this will move the origin of the 3D coordinate system
			// onto the plane
			p3d = p3d - p0;
			boost::geometry::point_2d p2d(p3d.dot(u), p3d.dot(v));
			boost::geometry::append(polygon,  p2d);
		}
	
	boost::geometry::correct(polygon);
	return boost::geometry::area(polygon);
	
    }
     /**Calculates the extension of a shape using the envelope
       @param alpha_ptr input point cloud
       @return max length of bounding rect.
    */
    
    double calcExtension(pcl::PointCloud<PointT>::ConstPtr bounding_ptr)
    {
	Eigen::Vector3f p0(bounding_ptr->points[0].x,
			   bounding_ptr->points[0].y,
			   bounding_ptr->points[0].z);
	Eigen::Vector3f p1(bounding_ptr->points[1].x,
			   bounding_ptr->points[1].y,
			   bounding_ptr->points[1].z);
	Eigen::Vector3f p2(bounding_ptr->points[2].x,
			   bounding_ptr->points[2].y,
			   bounding_ptr->points[2].z);
	
	Eigen::Vector3f d1=p0-p1;
	Eigen::Vector3f d2=p1-p2;

	return std::max(d1.norm(),d2.norm());
    }
  
    /**Calculates the center of gravity of a pointcloud using alpha shapes
       @param alpha_ptr input point cloud
       @return CoG
    */

    Eigen::Vector3f calcCoG(pcl::PointCloud<PointT>::ConstPtr cloud_ptr, pcl::ModelCoefficientsConstPtr coeff_plane)
    {

	boost::geometry::polygon_2d polygon; 
	Eigen::Vector3f plane_normal,v,u;
	Eigen::Vector3f cog_3d;
	
	assert (cloud_ptr!=NULL && coeff_plane!=NULL);
	plane_normal= Eigen::Vector3f(coeff_plane->values[0],
				      coeff_plane->values[1],
				      coeff_plane->values[2]);
	
	// compute an orthogonal normal to the plane normal	
	v = plane_normal.unitOrthogonal();
	
	// take the cross product of the two normals to get
	// a thirds normal, on the plane
	u = plane_normal.cross(v);
	
	// project the 3D point onto a 2D plane
	
	// choose a point on the plane
	Eigen::Vector3f p0(cloud_ptr->points[0].x,
			cloud_ptr->points[0].y,
			cloud_ptr->points[0].z);
    
	//project each point on a 2D plane for area estimation
	for(size_t i=0; i<cloud_ptr->points.size(); i++)
		{
			Eigen::Vector3f p3d(cloud_ptr->points[i].x,
					    cloud_ptr->points[i].y,
					    cloud_ptr->points[i].z);

			// subtract all 3D points with a point in the plane
			// this will move the origin of the 3D coordinate system
			// onto the plane
			p3d = p3d - p0;
			boost::geometry::point_2d p2d(p3d.dot(u), p3d.dot(v));
			boost::geometry::append(polygon,  p2d);
		}
	
	boost::geometry::correct(polygon);
	
	boost::geometry::point_2d cog_2d(0,0);
	boost::geometry::centroid(polygon, cog_2d);
		
	cog_3d=boost::geometry::get<0>(cog_2d)*u;
	cog_3d+=boost::geometry::get<1>(cog_2d)*v;
	cog_3d+=p0;	
	return cog_3d;
    }
    
   /**Displays the normal in RVIZ as a marker.
       @param cog
       @param normal
     */
    void showNormals(std::vector<Eigen::Vector3f> normals,std::vector<Eigen::Vector3f> cogs){
	
	visualization_msgs::MarkerArray normals_marker_array_msg;   

	normals_marker_array_msg.markers.resize(normals.size());
	for (int i=0;i<normals.size();i++){
	    
	    geometry_msgs::Point pos;
	    pos.x=cogs[i][0];
	    pos.y=cogs[i][1];
	    pos.z=cogs[i][2];
	    
	    normals_marker_array_msg.markers[i].pose.position = pos;
	    
	    //axis-angle rotation
	    btVector3 axis(normals[i][0],normals[i][1],normals[i][2]);
	    btVector3 marker_axis(1, 0, 0);
	    btQuaternion qt(marker_axis.cross(axis.normalize()), marker_axis.angle(axis.normalize()));
	    geometry_msgs::Quaternion quat_msg;
	    tf::quaternionTFToMsg(qt, quat_msg);
	    normals_marker_array_msg.markers[i].pose.orientation = quat_msg;
	    
	    normals_marker_array_msg.markers[i].header.frame_id = _frame_id;
	    normals_marker_array_msg.markers[i].header.stamp = ros::Time::now();
	    normals_marker_array_msg.markers[i].id = i;
	    normals_marker_array_msg.markers[i].ns = "Normals";
	    normals_marker_array_msg.markers[i].color.r = 1.0f;
	    normals_marker_array_msg.markers[i].color.g = 0.0f;
	    normals_marker_array_msg.markers[i].color.b = 0.0f;
	    normals_marker_array_msg.markers[i].color.a = 0.5f;
	    normals_marker_array_msg.markers[i].lifetime = ros::Duration();
	    normals_marker_array_msg.markers[i].type = visualization_msgs::Marker::ARROW;
	    normals_marker_array_msg.markers[i].scale.x = 0.2;
	    normals_marker_array_msg.markers[i].scale.y = 0.2;
	    normals_marker_array_msg.markers[i].scale.z = 0.2;

	    normals_marker_array_msg.markers[i].action = visualization_msgs::Marker::ADD;	    	    	    
	}

	_pub_normal_marker_array.publish(normals_marker_array_msg);


    }


  
private:

    ros::NodeHandle _n;
    ros::Publisher _pub_hull;
    ros::Publisher _pub_bounding;
    ros::Publisher _pub_alpha;
    ros::Publisher _pub_alpha_polygon;
    ros::Publisher _pub_cog;
    ros::Publisher _pub_shape_features;
    ros::Publisher _pub_shape_feature_bag;
    //    ros::Publisher _pub_normal_marker;
    ros::Publisher _pub_normal_marker_array;
    double _alpha_value;
    ros::Subscriber _sub; 
    std::string _frame_id;
    

};


int main (int argc, char** argv)
{
  ros::init (argc, argv, "feature_extractor");
  ros::NodeHandle n;
  FeatureExtractor extractor(n);
  ros::spin ();
  return 0;
}
