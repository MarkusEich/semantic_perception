#ifndef POINTCLOUDDETECTION_HPP
#define POINTCLOUDDETECTION_HPP

#include <stdlib.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>


#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <object_detection/Objects.hpp>
#include <segmentation/plane.h>
#include <vector>
#include <pcl/segmentation/region_growing.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointRGBT;

typedef boost::shared_ptr<object_detection::Cylinder> SharedCylinder;
typedef boost::shared_ptr<object_detection::Box> SharedBox;

typedef boost::shared_ptr<std::vector<pcl::PointCloud<PointRGBT> > > SharedPointClusters;


namespace object_detection
{

class PointCloudDetection
{
public:
    PointCloudDetection();

    void setPointcloud(pcl::PointCloud<PointT>::Ptr points);
    void setPointcloud(pcl::PointCloud<PointRGBT>::Ptr points);

    /**
      * @brief Returns current point cloud
      * @return shared pointer of the point cloud
      */
    pcl::PointCloud<PointRGBT>::Ptr getPointCloud(){return boost::make_shared<pcl::PointCloud<PointRGBT> >(pointcloud_);}
    /**
      * @brief Returns point clouds of cylinders
      * @return shared pointer of the point cloud
      */
    SharedPointClusters getCylindersPoints(){return boost::make_shared<std::vector<pcl::PointCloud<PointRGBT> > >(cylinder_clusters_);}
    /**
      * @brief Returns point clouds of boxes
      * @return shared pointer of the point cloud
      */
    SharedPointClusters getBoxesPoints(){return boost::make_shared<std::vector<pcl::PointCloud<PointRGBT> > >(box_clusters_);}

    void setBoxModel(const Box &boxModel){
        boxModel_.color=boxModel.color;
        boxModel_.dimensions=boxModel.dimensions;

        //norm box model; a largest edge, b middle edge, c smallest edge
        //dunno how to easily sort vector3f
        std::vector<double> tmp;
        tmp.push_back(boxModel_.dimensions[0]);
        tmp.push_back(boxModel_.dimensions[1]);
        tmp.push_back(boxModel_.dimensions[2]);
        std::sort(tmp.begin(),tmp.end());
        boxModel_.dimensions=base::Vector3d(tmp[2],tmp[1],tmp[0]);
        boxModelSet_=true;
    }

    void setCylinderModel(const Cylinder &cylinderModel){
        cylinderModel_.color=cylinderModel.color;
        cylinderModel_.diameter=cylinderModel.diameter;
        cylinderModel_.height=cylinderModel.height;
        cylinderModelSet_=true;        
    }

    void removeHorizontalPlanes(double minExtension, double maxHorizVariance);
    bool getCylinders(std::vector<object_detection::Cylinder>& cylinders);
    bool getBoxes(std::vector<object_detection::Box>& boxes);
    void applyPassthroughFilter(double x, double y, double z);
    void applyVoxelFilter(double voxel_size);
    void applyOutlierFilter(double radius);

    void setDebug(bool debug){debug_=debug;}

private:

    //private members

    pcl::PointCloud<PointRGBT> pointcloud_; //colored pointcloud    
    pcl::PointCloud<pcl::Normal> cloud_normals_; //Point cloud for storing the cloud_normals


    std::vector<pcl::PointCloud<PointRGBT> > cylinder_clusters_; //colored cylinder point cloud
    std::vector<pcl::PointCloud<PointRGBT> > box_clusters_; //colored cylinder point cloud

    object_detection::Cylinder cylinderModel_; //cylinder model for object search
    object_detection::Box boxModel_; //box model for object search

    bool debug_; //enable debug
    int ransac_iterations_;
    double ransac_dist_threshold_;
    double ransac_normal_dist_weight_;
    double segmentation_cluster_dist_;
    double smoothness_;
    double curvature_;
    int min_cluster_size_;
    int max_cluster_size_;
    double max_cluster_dist_;
    bool cylinderModelSet_;
    bool boxModelSet_;
    double cylinderModelThreshold_;
    double boxModelThreshold_;

    double ransac_min_radius_;
    double ransac_max_radius_;

private:

    //private functions
    //gets the match of a given Cylinder
    double getCylinderScore(object_detection::Cylinder detectedObject);
    //gets the match of a given Box
    double getBoxScore(object_detection::Box detectedObject);
    //computes normals of the pointcloud member variable
    bool computeNormals();
    //get clusters from the current point cloud
    void getClusters(boost::shared_ptr<std::vector<pcl::PointCloud<PointRGBT> > >& clusters);
    //gets the percentage of a point cloud beeing a cylinder
    double getCylinderLikelihood(
            const pcl::PointCloud<PointRGBT>::Ptr input_cloud_ptr, pcl::ModelCoefficients& coefficients);
    //gets the hight of the cylinder, given a point cloud and the coefficients
    double getCylinderHeight(
            const pcl::PointCloud<PointRGBT>::Ptr input_cloud_ptr, PointT& max_pt, PointT& min_pt, const pcl::ModelCoefficients coefficients);


};

}
#endif // POINTCLOUDDETECTION_HPP
