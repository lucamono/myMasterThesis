#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

class DatasetInterface {
 
    public:
	DatasetInterface(){};
	void updateDataset(int scenario, cv::Mat image_Reg_rgb, cv::Mat image_Reg_depth, cv::Mat image_NoReg_rgb, cv::Mat image_NoReg_depth, cv::Mat debug_image, Eigen::Vector3f eigenPoint3D, cv::Point posReg_xy, cv::Point posNoReg_xy, float depth, float pitch, float yaw, bool success);
	int getCurrentScenario();
	void updateResumeState(int scenario);
	int checkInputKey(char inp);
	int initDataset();
};
