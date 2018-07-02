#pragma once

#include <vector>
#include <string>
#include <functional>
#include <thread>

#include <opencv2/opencv.hpp>

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

namespace flexsight
{
class CameraDriver
{
public:

  CameraDriver();
  ~CameraDriver();

  void searchCameras();
  int numCameras(){ return num_cams_; };
  std::vector<std::string> cameraIDs(){ return cam_unique_ids_; };

  bool startAcquisition();
  bool startThreadedAcquisition();
  void stopAcquisition();

  std::vector<cv::Mat> getNextImages();

  void setThreadCallback( std::function<void(std::vector<cv::Mat> &)> &callback );

  double getExposureTime();
  double getGain();
  double getFrameRate();

  void getExposureTimeLimits( double &min_t_us, double &max_t_us );
  void getGainLimits( double &min_gain, double &max_gain );
  void getFrameRateLimits( double &min_fps, double &max_fps );

  void setExposureTime( double t_us );
  void setAutoExposure();
  void setGain( double gain );

  void setAutoGain();
  void setFrameRate( double fps );
  void setAutoFrameRate();

  void switchOnCameras();
  void switchOffCameras();

  bool enableSoftwareTrigger();
  bool enableHardwareTrigger();
  void disableTrigger();

  void softwareTrigger();

private:

  cv::Mat convert2OpenCV( Spinnaker::ImagePtr &img );
  bool startAcquisition( bool threaded );
  void threadedAcquisition();

  Spinnaker::SystemPtr system_ptr_;
  std::vector<Spinnaker::CameraPtr> cam_ptrs_;
  std::vector<std::string> cam_unique_ids_;
  bool cam_acq_flag_, threaded_acq_;
  std::thread cam_thread_;
  std::vector<Spinnaker::GenApi::CCommandPtr> cam_s_trigger_;

  int num_cams_;
  bool soft_trigger_enabled_;

  std::function<void(std::vector<cv::Mat> &)> imgCallack_;

};
};