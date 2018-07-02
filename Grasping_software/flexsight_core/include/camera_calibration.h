#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "cv_ext/cv_ext.h"

namespace flexsight
{
class CameraCalibration
{
public:

  void setFileList( std::vector<std::string> &f_list ){ files_list_ = f_list; };
  /** @brief Extract the chessboard corners from each image.
   *
   * @param[in] pyr_num_levels If greater than zero, build a gaussian pyramid of the image
   *                           and start to extract the corners from the higher
   *                           level of gaussian pyramid (useful for big images). */
  void extractImagePoints( int pyr_num_levels = 1, bool show_corners = true );

  /** Perform the calibration, providing the rms reprojection error. */
  double calibrate();
  /** Provide the resulting camera model. */
  cv_ext::PinholeCameraModel getCamModel();
  /** Set a previously computed camera model. */
  void setCamModel( cv_ext::PinholeCameraModel &model );

  void computeExtrinsicParameters( cv::Mat &r_vec, cv::Mat &t_vec );

  void showUndistorted();

  /** Set the size of the board, i.e. the number of internal corners by width and height */
  void setBoardSize( cv::Size s ){ board_size_ = s; };
  /** Set the size of a square in your defined unit (point, millimeter,etc) */
  void setSquareSize( float s ){ square_size_ = s; };
  /** If enabled, in the calibratin consider only fy as a free parameter, with fx/fy = 1 */
  void fixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };
  /** If enabled, the principal point is not changed during the global optimization*/
  void fixPrincipalPoint( bool enable ){fix_principal_point_ = enable; };
  /** If enabled, in the calibratin assume zero tangential distortion */
  void zeroTanDist( bool enable ){zero_tan_dist_ = enable; };

protected:

  cv::Size board_size_ = cv::Size(-1,-1);
  float square_size_ = 0.0f;
  bool fix_aspect_ratio_ = false;
  bool zero_tan_dist_ = false;
  bool fix_principal_point_ = false;

  cv::Size image_size_ = cv::Size(-1,-1);
  cv::Mat camera_matrix_, dist_coeffs_;

  std::vector<std::string> files_list_;
  std::vector<std::vector<cv::Point2f> > image_points_;
  std::vector<float> per_view_errors_;
};

void readExtrinsicsFromFile( std::string filename, cv::Mat &r_vec, cv::Mat &t_vec );
void writeExtrinsicsToFile( std::string filename, const cv::Mat &r_vec, const cv::Mat &t_vec );

}