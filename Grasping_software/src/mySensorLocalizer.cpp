#include "mySensorLocalizer.h"

    Eigen::Affine3d MySensorLocalizer::localizeBoardFromCamera(cv::Mat image_rgb)
    {
	cam_model.readFromFile("/home/luca/Scrivania/Grasping_software/sensorLocalizer/config/kinect2_camera_calib.yml");
	//get intrinsic of camera
	K = cam_model.cameraMatrix();
	//get distortion parameter of camera
	dist = cam_model.distorsionCoeff();
	
	cv::Mat mark = cv::imread("/home/luca/Scrivania/Grasping_software/sensorLocalizer/26696425_10215379129058310_1113899081_n.png", CV_LOAD_IMAGE_UNCHANGED);
	SensorLocalizer sl(mark);
	
	//set intrinsic parameters of camera
	sl.setIntrinsics(K,dist);
        Eigen::Affine3d board_pose;
	cv::Mat detection_output;
	Eigen::Affine3d a;
	a.setIdentity();
	//add measured offset from board
	a.translation()(0)=-0.339;
	a.translation()(1)=-0.092;
	bool res = sl.localizeBoard(image_rgb,a,board_pose,detection_output);
	//cv::imshow("detection_output",detection_output);
	//cv::waitKey(0);
	return board_pose;
    }
    
    Eigen::Vector3f MySensorLocalizer::pixelTo3DWorldPoint(cv::Point2f xy_grasping_coords, float real_depth, Eigen::Affine3d board_pose)
    {
	Eigen::Vector3f eigenPoint3D;
	float point2D[2];
	point2D[0]=xy_grasping_coords.x;
	point2D[1]=xy_grasping_coords.y;
	float point3D[3];
	cv_ext::PinholeCameraModel cam_model = getCameraModel();
	cam_model.unproject(point2D,real_depth,point3D);
	eigenPoint3D(0)=point3D[0];
	eigenPoint3D(1)=point3D[1];
	eigenPoint3D(2)=point3D[2];
	Eigen::Affine3f bp(board_pose);
	bp.translation() = bp.translation()*1000.0;
	std::cout << board_pose.matrix() << std::endl;
        eigenPoint3D = bp.inverse()*eigenPoint3D;
	return eigenPoint3D;
    }
