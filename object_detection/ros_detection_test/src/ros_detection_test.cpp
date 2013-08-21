#include <ros/ros.h>
#include <object_detection/PointCloudDetection.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//use new header for messages. New in PCL 1.7
#include <pcl_conversions/pcl_conversions.h>


/**
 *Test for object detection
 */

using namespace object_detection;

class Tester
{

public:

    Tester(ros::NodeHandle nh):nh_(nh){
        pub_=nh_.advertise<sensor_msgs::PointCloud2>("cloud",10);

    }
    ~Tester(){}

    bool run(){

         PointCloudDetection detector;
         pcl::PointCloud<pcl::PointXYZ> cloud;
         pcl::PCLPointCloud2 out_msg;

         std::vector<Cylinder> cylinders;
         std::vector<Box> boxes;


         if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../cloud_data/object_scans_10.pcd", cloud) == -1) //* load the file
         {
           PCL_ERROR ("Couldn't read test file\n");
           return false;
         }
         std::cout << "Loaded " << cloud.width * cloud.height << " data points from test_pcd.pcd with the following fields: " << std::endl;


         detector.applyOutlierFilter(0.10);
         detector.applyPassthroughFilter(2.0,1.0,1.0);
         detector.applyVoxelFilter(0.01);
         //remove all planes which are larger 50cm and 0.1 radian to a horizontal plane
         detector.removeHorizontalPlanes(0.5,0.30);

         Cylinder cylinder_model;

         cylinder_model.diameter=0.08;
         cylinder_model.height=0.12;
         cylinder_model.color=Color(255,255,0);

         Box box_model;
         box_model.dimensions=Eigen::Vector3d(0.20,0.10,0.04);
         box_model.color=Color(0,0,255);

         detector.setCylinderModel(cylinder_model);
         detector.setBoxModel(box_model);

         if (detector.getCylinders(cylinders)){
             std::vector<Cylinder>::iterator iterC=cylinders.begin();
             for (;iterC!=cylinders.end();iterC++){
                 std::cout<<"Cylinder Diameter: "<<iterC->diameter<<std::endl;
                 std::cout<<"Cylinder Height: "<<iterC->height<<std::endl;
                 std::cout<<"Cylinder Likelihood: "<<iterC->likelihood<<std::endl;
                 std::cout<<"Cylinder Position: "<<std::endl;
                 std::cout<<iterC->position<<std::endl;
                 std::cout<<"Cylinder Orientation: (Quaternion): "<<iterC->orientation.x()<<","
                                                             <<iterC->orientation.y()<<","
                                                             <<iterC->orientation.z()<<","
                                                             <<iterC->orientation.w()<<std::endl;
             }
         }

         if (detector.getBoxes(boxes)){
             std::vector<Box>::iterator iterB=boxes.begin();
             for (;iterB!=boxes.end();iterB++){
                 std::cout<<"Box Dimension: "<<iterB->dimensions<<std::endl;
                 std::cout<<"Box Likelihood: "<<iterB->likelihood<<std::endl;
                 std::cout<<"Box Position: "<<std::endl;
                 std::cout<<iterB->position<<std::endl;
                 std::cout<<"Box Orientation: (Quaternion): "<<iterB->orientation.x()<<","
                                                             <<iterB->orientation.y()<<","
                                                             <<iterB->orientation.z()<<","
                                                             <<iterB->orientation.w()<<std::endl;

             }
         }



         pcl::toPCLPointCloud2(*detector.getPointCloud(),out_msg);


         //out_msg.header.stamp=ros::Time::now();
         out_msg.header.frame_id="/base_link";
         //pub_.publish(out_msg);




         return true;
    }


private:



    ros::NodeHandle nh_;
    ros::Publisher pub_;

};

int main(int argc, char** argv){

    ros::init(argc,argv,"ros_detection_test");
    ros::NodeHandle nh;
    Tester tester(nh);
    if (tester.run()) std::cout<<"success!"<<std::endl;
    else
        std::cout<<"fail!"<<std::endl;
    
    return 0;
}



