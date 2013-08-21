#include <boost/test/unit_test.hpp>

#include <object_detection/PointCloudDetection.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace object_detection;

BOOST_AUTO_TEST_CASE( detect_in_pointcloud )
{
    // load a point cloud and check for objects
    
    pcl::PointCloud<pcl::PointXYZ> cloud;
    PointCloudDetection detector;

    Cylinder cylinder_model;
    Box box_model;

    std::vector<Cylinder> cylinders;
    std::vector<Box> boxes;


    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test/cloud_data/object_scans_10.pcd", cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read test file\n");
    }
    else{
         std::cout << "Loaded " << cloud.width * cloud.height << " data points from test_pcd.pcd with the following fields: " << std::endl;


         detector.setPointcloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cloud));
         detector.setDebug(true);
         detector.applyOutlierFilter(0.10);
         detector.applyPassthroughFilter(2.0,1.0,1.0);
         detector.applyVoxelFilter(0.01);
         //remove all planes which are larger 50cm and 0.1 radian to a horizontal plane
         detector.removeHorizontalPlanes(0.5,0.30);

         cylinder_model.diameter=0.08;
         cylinder_model.height=0.12;
         cylinder_model.color=Color(255,255,0);

         box_model.dimensions=Eigen::Vector3d(0.20,0.10,0.04);
         box_model.color=Color(0,0,255);

         detector.setCylinderModel(cylinder_model);
         detector.setBoxModel(box_model);

         //show extracted objects
         pcl::visualization::PCLVisualizer viewer ("Object viewer");
         viewer.setBackgroundColor(0, 0, 0);
         viewer.addPointCloud(detector.getPointCloud(),"cloud");
         viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
         viewer.addCoordinateSystem (1.0);
         viewer.initCameraParameters ();


         if (detector.getCylinders(cylinders)){
             std::vector<Cylinder>::iterator iterC=cylinders.begin();
             for (;iterC!=cylinders.end();iterC++){
                 std::cout<<"Cylinder Diameter: "<<iterC->diameter<<std::endl;
                 std::cout<<"Cylinder Height: "<<iterC->height<<std::endl;
                 std::cout<<"Cylinder Likelihood: "<<iterC->likelihood<<std::endl;
                 std::cout<<"Cylinder Position: "<<endl;
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


        //get points from cylinders and display
        int id=0;
        SharedPointClusters cylinderClusters;
        cylinderClusters=detector.getCylindersPoints();

        std::vector<pcl::PointCloud<PointRGBT> >::iterator clusterIterator=cylinderClusters->begin();

        for(;clusterIterator!=cylinderClusters->end();clusterIterator++){

         std::stringstream ss;
         ss << "cylinder";
         ss << id;
         viewer.addPointCloud(boost::make_shared<pcl::PointCloud<PointRGBT> >(*clusterIterator),ss.str());
         viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
         viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0,1.0,0.0, ss.str());
         id++;
        }

        //get points from boxes and display
        id=0;
        SharedPointClusters boxClusters;
        boxClusters=detector.getBoxesPoints();
        clusterIterator=boxClusters->begin();

        for(;clusterIterator!=boxClusters->end();clusterIterator++){
            std::stringstream ss;
            ss << "box";
            ss << id;
            viewer.addPointCloud(boost::make_shared<pcl::PointCloud<PointRGBT> >(*clusterIterator),ss.str());
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0,0.0,1.0, ss.str());
            id++;
        }

        while(!viewer.wasStopped())
        {
            viewer.spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }


    }

}
