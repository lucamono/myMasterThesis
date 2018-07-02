#include "Kinect2Interface.h"
#include <pcl/visualization/cloud_viewer.h>
#include "sensor_localizer.h"


using namespace std;


int main(int argc, char *argv[])
{
  
  processor freenectprocessor = OPENGL;
  bool mirror_frame = true;
  
  Kinect2Interface k2_interface(freenectprocessor, mirror_frame);
  Kineckt2Frame acquisition;
  //get acquisition frame
  k2_interface.getKinect2Frame(acquisition, false);
  cv::Mat image=acquisition.rgb_img;
  cvtColor(image, image, CV_BGRA2BGR);
  cv::imshow("rgb",image);
  //save image on file
  //cv::imwrite("kinect_image.jpg",image);
  cv::waitKey(0);
  k2_interface.shutDown();
  
  cv_ext:: PinholeCameraModel cam_model;
  cam_model.readFromFile("/home/luca/Scrivania/sensorLocalizer/config/kinect2_camera_calib.yml");
  cv::Mat K = cam_model.cameraMatrix();
  cv::Mat dist = cam_model.distorsionCoeff();
  
  //get intrinsic Kinect parameters
//   Kinect2Params kinect_params = k2_interface.getKinect2InternalParams();
//   cv::Mat K = (cv::Mat_<double>(3,3) << kinect_params.rgb_params.fx, 0, kinect_params.rgb_params.cx,
// 	                        0, kinect_params.rgb_params.fy, kinect_params.rgb_params.cy,
// 	                        0, 0, 1);
//   cv::Mat dist = (cv::Mat_<double>(5,1) << 0,0,0,0,0);
 
        
  //instantiate Sensor Localizer 
  cv::Mat mark = cv::imread("/home/luca/Scrivania/sensorLocalizer/26696425_10215379129058310_1113899081_n.png", CV_LOAD_IMAGE_UNCHANGED);
  SensorLocalizer sl(mark);
 
  //set intrinsic parameters of camera
  sl.setIntrinsics(K,dist);
  
  cv::Mat detection_output;
  Eigen::Affine3d board_pose;
  Eigen::Affine3d a;
  a.setIdentity();
  a.translation()(0)=-0.339;
  a.translation()(1)=-0.092;
  bool res = sl.localizeBoard(image,a,board_pose,detection_output);
   std::cout << board_pose.matrix() << std::endl;
  cv::imshow("detection_output",detection_output);
  
  cv::waitKey(0);
  cvDestroyAllWindows();
  return 0;
}

