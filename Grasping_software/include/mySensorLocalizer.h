#include "sensor_localizer.h"

class MySensorLocalizer {
    
    private:
	cv_ext::PinholeCameraModel cam_model;
	cv::Mat K;
	cv::Mat dist;
  
    public:
	MySensorLocalizer(){};
	cv_ext::PinholeCameraModel getCameraModel(){ return this->cam_model;}
	Eigen::Affine3d localizeBoardFromCamera(cv::Mat image_rgb);
	Eigen::Vector3f pixelTo3DWorldPoint(cv::Point2f xy_grasping_coords, float real_depth, Eigen::Affine3d board_pose);
	
};
