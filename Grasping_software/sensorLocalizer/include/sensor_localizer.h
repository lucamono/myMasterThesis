#pragma once

#include <opencv2/opencv.hpp>
#include <cv_ext/cv_ext.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/aruco.hpp>

class SensorLocalizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SensorLocalizer(const cv::Mat& meta_board_image);
  
  bool localizeBoard(const cv::Mat& im,  Eigen::Affine3d ref_sys_transform, Eigen::Affine3d& board_pose, cv::Mat & detection_output, bool reload_board = false);
  bool localizeCheckerboard(const cv::Mat& im, Eigen::Affine3d& board_pose);
  
  void setIntrinsics(const cv::Mat& K, const cv::Mat& Dist)
  {
    K_=K.clone(); Dist_coeff_=Dist.clone();
  }
  
  void computeMasks(const Eigen::Affine3d& camera_pose, const cv::Size& im_size,
                    cv::Mat& board_mask, cv::Mat& ws_mask);
  
private:
  
  cv::Ptr<cv::aruco::DetectorParameters> detectorParams_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  cv::Ptr<cv::aruco::Board> board_;
  
  cv::Mat K_;
  cv::Mat Dist_coeff_;
};
