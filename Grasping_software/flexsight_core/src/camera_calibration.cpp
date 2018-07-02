#include "camera_calibration.h"

#include "cv_ext/cv_ext.h"

#include <iostream>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv_ext;

namespace flexsight
{

static void calcBoardCornerPositions ( vector<Point3f>& object_points, Size board_size, float square_size )
{
  for ( int i = 0; i < board_size.height; ++i )
    for ( int j = 0; j < board_size.width; ++j )
      object_points.push_back ( Point3f ( float ( j*square_size ), float ( i*square_size ), 0 ) );
}

static bool findChessboardCornersPyr( Mat img, Size board_size, vector<Point2f> &img_pts,
                                      int pyr_num_levels )
{
  bool corners_found = false;
  ImagePyramid img_pyr;
  img_pyr.buildGuassianPyrFromImage(img, pyr_num_levels, -1, false );
  if( findChessboardCorners ( img_pyr[pyr_num_levels - 1], board_size, img_pts,
                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE ) )
  {
    corners_found = true;

    // Look for the white circle in the pattern corresponding to the origin of the reference frame
    vector<Point2f> square_pts(4), square_img_pts(4), test_pts(5), test_img_pts(5);

    square_pts[0] = Point2f(0.0f, 0.0f);
    square_pts[1] = Point2f(1.0f, 0.0f);
    square_pts[2] = Point2f(0.0f, 1.0f);
    square_pts[3] = Point2f(1.0f, 1.0f);

    square_img_pts[0] = img_pts[0];
    square_img_pts[1] = img_pts[1];
    square_img_pts[2] = img_pts[board_size.width];
    square_img_pts[3] = img_pts[board_size.width + 1];

    Mat h1 = getPerspectiveTransform(square_pts, square_img_pts);

    int n_pts = img_pts.size();
    square_img_pts[0] = img_pts[n_pts - board_size.width - 2];
    square_img_pts[1] = img_pts[n_pts - board_size.width - 1];
    square_img_pts[2] = img_pts[n_pts - 2];
    square_img_pts[3] = img_pts[n_pts - 1];

    Mat h2 = getPerspectiveTransform(square_pts, square_img_pts);

    test_pts[0] = Point2f(0.5f, 0.5f);
    test_pts[1] = Point2f(0.6f, 0.5f);
    test_pts[2] = Point2f(0.4f, 0.5f);
    test_pts[3] = Point2f(0.5f, 0.6f);
    test_pts[4] = Point2f(0.5f, 0.4f);

    Mat debug_img;
    cvtColor(img_pyr[pyr_num_levels - 1], debug_img, cv::COLOR_GRAY2BGR);

    perspectiveTransform(test_pts, test_img_pts, h1);

    float score1 = 0.0f, score2 = 0.0f;

    for( auto &p : test_img_pts )
      score1 += img_pyr[pyr_num_levels - 1].at<uchar>(p.y, p.x);

    perspectiveTransform(test_pts, test_img_pts, h2);

    for( auto &p : test_img_pts )
      score2 += img_pyr[pyr_num_levels - 1].at<uchar>(p.y, p.x);

    if( score2 > score1 )
      std::reverse(img_pts.begin(),img_pts.end());

    cornerSubPix ( img_pyr[pyr_num_levels - 1], img_pts, Size ( 11,11 ),
                    Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );

    for( int l = pyr_num_levels - 2; l >= 0; l-- )
    {
      for( auto &p : img_pts )
        p *= 2.0f;
      cornerSubPix ( img_pyr[l], img_pts, Size ( 11,11 ),
                    Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );
    }
  }
  return corners_found;
}

void CameraCalibration::extractImagePoints( int pyr_num_levels, bool show_corners )
{
  if( files_list_.empty() || board_size_.width < 2 || board_size_.height < 2)
    return;

  image_points_.clear();
  image_points_.reserve(files_list_.size());

  for( auto &filename : files_list_ )
  {
    Mat img = imread(filename, cv::IMREAD_COLOR);
    if( !img.empty() )
    {
      image_size_ = img.size();
      Mat img_gray;
      cvtColor ( img, img_gray, cv::COLOR_BGR2GRAY );

      vector<Point2f> img_pts;
      if( findChessboardCornersPyr( img_gray, board_size_, img_pts, pyr_num_levels) )
      {
        image_points_.push_back ( img_pts );
        drawChessboardCorners ( img, board_size_, Mat ( img_pts ), true );
        if( show_corners )
          showImage(img, "Corners");
      }
    }
  }
}

double CameraCalibration::calibrate()
{
  camera_matrix_ = Mat::eye ( 3, 3, CV_64F );
  dist_coeffs_ = Mat::zeros ( 8, 1, CV_64F );

  vector<vector<Point3f> > object_points(1);
  calcBoardCornerPositions ( object_points[0], board_size_, square_size_ );
  object_points.resize ( image_points_.size(), object_points[0] );

  int flag = 0;
  if ( fix_principal_point_ ) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
  if ( zero_tan_dist_ ) flag |= cv::CALIB_ZERO_TANGENT_DIST;
  if ( fix_aspect_ratio_ ) flag |= cv::CALIB_FIX_ASPECT_RATIO;

  vector<Mat> r_vecs, t_vecs;

  double rms = cv::calibrateCamera ( object_points, image_points_, image_size_, camera_matrix_,
                                     dist_coeffs_, r_vecs, t_vecs,
                                     flag|cv::CALIB_FIX_K4|cv::CALIB_FIX_K5|cv::CALIB_FIX_K6 );

  // Compute reprojection error
  vector<Point2f> rep_image_points;
  int i, num_points = 0;
  double total_err = 0, err;
  per_view_errors_.resize ( object_points.size() );

  for ( i = 0; i < ( int ) object_points.size(); ++i )
  {
    projectPoints ( Mat ( object_points[i] ), r_vecs[i], t_vecs[i], camera_matrix_,
                    dist_coeffs_, rep_image_points );
    err = norm ( Mat ( image_points_[i] ), Mat ( rep_image_points ), NORM_L2 );

    int n = ( int ) object_points[i].size();
    per_view_errors_[i] = ( float ) std::sqrt ( err*err/n );
    total_err += err*err;
    num_points += n;
  }

  return std::sqrt ( total_err/num_points );
}

PinholeCameraModel CameraCalibration::getCamModel()
{
  return PinholeCameraModel( camera_matrix_, image_size_.width,
                             image_size_.height, 1.0, dist_coeffs_ );
}

void CameraCalibration::setCamModel( cv_ext::PinholeCameraModel &model )
{
  camera_matrix_ = model.cameraMatrix();
  dist_coeffs_ = model.distorsionCoeff();
  image_size_ = model.imgSize();
}

void CameraCalibration::computeExtrinsicParameters( cv::Mat &r_vec, cv::Mat &t_vec )
{
  vector<Point3f> object_points, corner_pos;
  vector<Point2f> image_points;

  calcBoardCornerPositions ( corner_pos, board_size_, square_size_ );
  object_points.reserve(corner_pos.size()*image_points_.size());
  image_points.reserve(corner_pos.size()*image_points_.size());
  for( int i = 0; i < image_points_.size(); i++ )
  {
    object_points.insert(object_points.end(), corner_pos.begin(), corner_pos.end());
    image_points.insert(image_points.end(), image_points_[i].begin(), image_points_[i].end());
  }

  solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_, r_vec, t_vec );
}

void CameraCalibration::showUndistorted()
{
  for( auto &filename : files_list_ )
  {
    Mat img = imread(filename), und_img;
    if( !img.empty() )
    {
      und_img = Mat(img.size(), img.type() );
      cv::undistort ( img, und_img, camera_matrix_, dist_coeffs_ );
      showImage(img, "Undistorted image");
    }
  }
}

void readExtrinsicsFromFile( std::string filename, cv::Mat &r_vec, cv::Mat &t_vec )
{
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  std::string ext_str1 = filename.substr ((pos1 > 0)?pos1:0);
  std::string ext_str2 = filename.substr ((pos2 > 0)?pos2:0);

  if( ext_str1.compare(".yml") && ext_str2.compare(".yaml") )
    filename += ".yml";

  FileStorage fs(filename, FileStorage::READ);

  fs["r_vec"] >> r_vec;
  fs["t_vec"] >> t_vec;

  fs.release();
}

void writeExtrinsicsToFile( std::string filename, const cv::Mat &r_vec, const cv::Mat &t_vec )
{
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  std::string ext_str1 = filename.substr ((pos1 > 0)?pos1:0);
  std::string ext_str2 = filename.substr ((pos2 > 0)?pos2:0);

  if( ext_str1.compare(".yml") && ext_str2.compare(".yaml") )
    filename += ".yml";

  FileStorage fs(filename, FileStorage::WRITE);

  fs << "r_vec" << r_vec;
  fs << "t_vec" << t_vec;

  fs.release();
}

}

#if 0

bool runCalibrationAndSave ( Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,
                             vector<vector<Point2f> > imagePoints );

int main ( int argc, char* argv[] )
{

  vector<vector<Point2f> > image_points_;
  Mat cameraMatrix, distCoeffs;
  Size imageSize;
  int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
  clock_t prevTimestamp = 0;
  const Scalar RED ( 0,0,255 ), GREEN ( 0,255,0 );
  const char ESC_KEY = 27;

  for ( int i = 0; ; ++i )
  {
    Mat view;
    bool blinkOutput = false;

    Mat tmp = s.nextImage();
    if(tmp.empty())
    {
      cout<<"Empty: "<<i<<endl;
      continue;
    }
    cv::resize(tmp, view, Size(), 0.5,0.5);

    if ( view.empty() || i == s.images_list_.size() - 1)       // If no more images then run calibration, save and stop loop.
    {
      if ( image_points_.size() > 0 )
        runCalibrationAndSave ( s, imageSize,  cameraMatrix, distCoeffs, image_points_ );
      break;
    }


    imageSize = view.size();  // Format input image.
    if ( s.flipVertical )    flip ( view, view, 0 );

    vector<Point2f> pointBuf;

    bool found;
    switch ( s.calibrationPattern ) // Find feature points on the input format
    {
    case Settings::CHESSBOARD:
      found = findChessboardCorners ( view, s.board_size_, pointBuf,
                                      CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE );
      break;
    case Settings::CIRCLES_GRID:
      found = findCirclesGrid ( view, s.board_size_, pointBuf );
      break;
    case Settings::ASYMMETRIC_CIRCLES_GRID:
      found = findCirclesGrid ( view, s.board_size_, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
      break;
    default:
      found = false;
      break;
    }

    if ( found )               // If done with success,
    {
      // improve the found corners' coordinate accuracy for chessboard
      if ( s.calibrationPattern == Settings::CHESSBOARD )
      {
        Mat viewGray;
        cvtColor ( view, viewGray, COLOR_BGR2GRAY );
        cornerSubPix ( viewGray, pointBuf, Size ( 11,11 ),
                       Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );
      }

      if ( mode == CAPTURING && // For camera only take new samples after delay time
           ( !s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC ) )
      {
        image_points_.push_back ( pointBuf );
        prevTimestamp = clock();
        blinkOutput = s.inputCapture.isOpened();
      }

      // Draw the corners.
      drawChessboardCorners ( view, s.board_size_, Mat ( pointBuf ), found );
//       imshow("corners",view);
//       waitKey(1000);
    }


    //----------------------------- Output Text ------------------------------------------------
    string msg = ( mode == CAPTURING ) ? "100/100" :
                 mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
    int baseLine = 0;
    Size textSize = getTextSize ( msg, 1, 1, 1, &baseLine );
    Point textOrigin ( view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10 );

    if ( mode == CAPTURING )
    {
      if ( s.showUndistorsed )
        msg = format ( "%d/%d Undist", ( int ) image_points_.size(), s.nrFrames );
      else
        msg = format ( "%d/%d", ( int ) image_points_.size(), s.nrFrames );
    }

    putText ( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED );

    if ( blinkOutput )
      bitwise_not ( view, view );

    //------------------------- Video capture  output  undistorted ------------------------------
    if ( mode == CALIBRATED && s.showUndistorsed )
    {
      Mat temp = view.clone();
      undistort ( temp, view, cameraMatrix, distCoeffs );
    }

    //------------------------------ Show image and check for input commands -------------------
    imshow ( "Image View", view );
    char key = ( char ) waitKey ( s.inputCapture.isOpened() ? 50 : s.delay );

    if ( key  == ESC_KEY )
      break;

    if ( key == 'u' && mode == CALIBRATED )
      s.showUndistorsed = !s.showUndistorsed;

    if ( s.inputCapture.isOpened() && key == 'g' )
    {
      mode = CAPTURING;
      image_points_.clear();
    }
  }

  // -----------------------Show the undistorted image for the image list ------------------------
  if ( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
  {
    Mat view, rview, map1, map2;
    initUndistortRectifyMap ( cameraMatrix, distCoeffs, Mat(),
                              getOptimalNewCameraMatrix ( cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0 ),
                              imageSize, CV_16SC2, map1, map2 );

    for ( int i = 0; i < ( int ) s.images_list_.size(); i++ )
    {
      view = imread ( s.images_list_[i], 1 );
      if ( view.empty() )
        continue;
      remap ( view, rview, map1, map2, INTER_LINEAR );
      imshow ( "Image View", rview );
      char c = ( char ) waitKey();
      if ( c  == ESC_KEY || c == 'q' || c == 'Q' )
        break;
    }
  }


  return 0;
}

static double computeReprojectionErrors ( const vector<vector<Point3f> >& objectPoints,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
    const Mat& cameraMatrix , const Mat& distCoeffs,
    vector<float>& perViewErrors )
{
  vector<Point2f> imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize ( objectPoints.size() );

  for ( i = 0; i < ( int ) objectPoints.size(); ++i )
  {
    projectPoints ( Mat ( objectPoints[i] ), rvecs[i], tvecs[i], cameraMatrix,
                    distCoeffs, imagePoints2 );
    err = norm ( Mat ( imagePoints[i] ), Mat ( imagePoints2 ), NORM_L2 );

    int n = ( int ) objectPoints[i].size();
    perViewErrors[i] = ( float ) std::sqrt ( err*err/n );
    totalErr        += err*err;
    totalPoints     += n;
  }

  return std::sqrt ( totalErr/totalPoints );
}

static void calcBoardCornerPositions ( Size boardSize, float squareSize, vector<Point3f>& corners,
                                       Settings::Pattern patternType /*= Settings::CHESSBOARD*/ )
{
  corners.clear();

  switch ( patternType )
  {
  case Settings::CHESSBOARD:
  case Settings::CIRCLES_GRID:
    for ( int i = 0; i < boardSize.height; ++i )
      for ( int j = 0; j < boardSize.width; ++j )
        corners.push_back ( Point3f ( float ( j*squareSize ), float ( i*squareSize ), 0 ) );
    break;

  case Settings::ASYMMETRIC_CIRCLES_GRID:
    for ( int i = 0; i < boardSize.height; i++ )
      for ( int j = 0; j < boardSize.width; j++ )
        corners.push_back ( Point3f ( float ( ( 2*j + i % 2 ) *squareSize ), float ( i*squareSize ), 0 ) );
    break;
  default:
    break;
  }
}

static bool runCalibration ( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                             vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                             vector<float>& reprojErrs,  double& totalAvgErr )
{

  cameraMatrix = Mat::eye ( 3, 3, CV_64F );
  if ( s.flag & CALIB_FIX_ASPECT_RATIO )
    cameraMatrix.at<double> ( 0,0 ) = 1.0;

  distCoeffs = Mat::zeros ( 8, 1, CV_64F );

  vector<vector<Point3f> > objectPoints ( 1 );
  calcBoardCornerPositions ( s.board_size_, s.square_size_, objectPoints[0], s.calibrationPattern );

  objectPoints.resize ( imagePoints.size(),objectPoints[0] );

  //Find intrinsic and extrinsic camera parameters
  double rms = calibrateCamera ( objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag|CALIB_FIX_K4|CALIB_FIX_K5 );

  cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

  bool ok = checkRange ( cameraMatrix ) && checkRange ( distCoeffs );

  totalAvgErr = computeReprojectionErrors ( objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs );

  return ok;
}

// Print camera parameters to the output file
static void saveCameraParams ( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                               const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                               const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                               double totalAvgErr )
{
  FileStorage fs ( s.outputFileName, FileStorage::WRITE );

  time_t tm;
  time ( &tm );
  struct tm *t2 = localtime ( &tm );
  char buf[1024];
  strftime ( buf, sizeof ( buf )-1, "%c", t2 );

  fs << "calibration_Time" << buf;

  if ( !rvecs.empty() || !reprojErrs.empty() )
    fs << "nrOfFrames" << ( int ) std::max ( rvecs.size(), reprojErrs.size() );
  fs << "image_Width" << imageSize.width;
  fs << "image_Height" << imageSize.height;
  fs << "board_Width" << s.board_size_.width;
  fs << "board_Height" << s.board_size_.height;
  fs << "square_Size" << s.square_size_;

  if ( s.flag & CALIB_FIX_ASPECT_RATIO )
    fs << "FixAspectRatio" << s.fix_aspect_ratio_;

  if ( s.flag )
  {
    sprintf ( buf, "flags: %s%s%s%s",
              s.flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
              s.flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
              s.flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
              s.flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "" );
    //cvWriteComment( *fs, buf, 0 );

  }

  fs << "flagValue" << s.flag;

  fs << "Camera_Matrix" << cameraMatrix;
  fs << "Distortion_Coefficients" << distCoeffs;

  fs << "Avg_Reprojection_Error" << totalAvgErr;
  if ( !reprojErrs.empty() )
    fs << "Per_View_Reprojection_Errors" << Mat ( reprojErrs );

  if ( !rvecs.empty() && !tvecs.empty() )
  {
    CV_Assert ( rvecs[0].type() == tvecs[0].type() );
    Mat bigmat ( ( int ) rvecs.size(), 6, rvecs[0].type() );
    for ( int i = 0; i < ( int ) rvecs.size(); i++ )
    {
      Mat r = bigmat ( Range ( i, i+1 ), Range ( 0,3 ) );
      Mat t = bigmat ( Range ( i, i+1 ), Range ( 3,6 ) );

      CV_Assert ( rvecs[i].rows == 3 && rvecs[i].cols == 1 );
      CV_Assert ( tvecs[i].rows == 3 && tvecs[i].cols == 1 );
      //*.t() is MatExpr (not Mat) so we can use assignment operator
      r = rvecs[i].t();
      t = tvecs[i].t();
    }
    //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
    fs << "Extrinsic_Parameters" << bigmat;
  }

  if ( !imagePoints.empty() )
  {
    Mat imagePtMat ( ( int ) imagePoints.size(), ( int ) imagePoints[0].size(), CV_32FC2 );
    for ( int i = 0; i < ( int ) imagePoints.size(); i++ )
    {
      Mat r = imagePtMat.row ( i ).reshape ( 2, imagePtMat.cols );
      Mat imgpti ( imagePoints[i] );
      imgpti.copyTo ( r );
    }
    fs << "Image_points" << imagePtMat;
  }
}

bool runCalibrationAndSave ( Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,vector<vector<Point2f> > imagePoints )
{
  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;
  double totalAvgErr = 0;

  bool ok = runCalibration ( s,imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                             reprojErrs, totalAvgErr );
  cout << ( ok ? "Calibration succeeded" : "Calibration failed" )
       << ". avg re projection error = "  << totalAvgErr ;

  if ( ok )
    saveCameraParams ( s, imageSize, cameraMatrix, distCoeffs, rvecs ,tvecs, reprojErrs,
                       imagePoints, totalAvgErr );
  return ok;
}
#endif