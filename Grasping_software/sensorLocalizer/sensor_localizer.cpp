#include "sensor_localizer.h"

using namespace std;
using namespace cv;

SensorLocalizer::SensorLocalizer(const cv::Mat& meta_board_image)
{
  int markersX;
  int markersY;
  float markerLength =0.0425;// 100;
  float markerSeparation =0.0067;// 16;
  int dictionaryId = 3;


  detectorParams_ = aruco::DetectorParameters::create();

  detectorParams_->adaptiveThreshWinSizeMax=23;
  detectorParams_->adaptiveThreshWinSizeStep=2;
  detectorParams_->adaptiveThreshConstant=1;
//   detectorParams_->minMarkerPerimeterRate = 0.01;
//   detectorParams_->maxMarkerPerimeterRate = 4;
// //   detectorParams_->adaptiveThreshWinSizeMax=50;
//   detectorParams_->adaptiveThreshWinSizeStep=5;  
//   detectorParams_->minCornerDistanceRate=.2; 
  detectorParams_->minMarkerDistanceRate=0.01; 
  detectorParams_->minDistanceToBorder=3;
//   detectorParams_->polygonalApproxAccuracyRate=1;
//  detectorParams_->doCornerRefinement = true;
  detectorParams_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  detectorParams_->maxErroneousBitsInBorderRate=.0;
//   detectorParams_->cornerRefinementMaxIterations = 160;
//   detectorParams_->cornerRefinementMinAccuracy = 1000;
//   detectorParams_->minOtsuStdDev=0;
//   detectorParams_->errorCorrectionRate=0;
//   detectorParams_->perspectiveRemovePixelPerCell=8;
//   detectorParams_->perspectiveRemoveIgnoredMarginPerCell=.3;

  dictionary_ =
      aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

  cv::Mat im_in=meta_board_image;
  markersX=im_in.cols;
  markersY=im_in.rows;

  // create board object
  Ptr<aruco::GridBoard> gridboard =
      aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary_);
  board_ = gridboard.staticCast<aruco::Board>();

  std::vector< std::vector< Point3f > > temp;
  std::vector< int > temp_ids;
  int iter=0;
  for(int r=0; r<im_in.rows; r++)
    for(int c=0; c<im_in.cols; c++)
    {
      if(im_in.at<uchar>(r,c)==0)
      {
        temp.push_back(board_->objPoints[iter]);
        temp_ids.push_back(board_->ids[iter]);
      }
      iter++;
    }

  board_->objPoints=temp;
  board_->ids=temp_ids;
  
}

void SensorLocalizer::computeMasks(const Eigen::Affine3d& camera_pose, const Size& im_size, Mat& board_mask, Mat& ws_mask)
{
  board_mask=cv::Mat(im_size, CV_8UC1, Scalar(0));
  ws_mask=cv::Mat(im_size, CV_8UC1, Scalar(0));
  
  double plus=.005;
  vector<Point3d> board_points(4);
  board_points[0]=Point3d(-0.002-plus,0-plus,0);
  board_points[1]=Point3d(1.2+plus,0-plus,0);
  board_points[2]=Point3d(1.2+plus,0.595+plus,0);
  board_points[3]=Point3d(-.002-plus,0.595+plus,0);
  
  vector<Point3d> ws_points(4);
  ws_points[0]=Point3d(0.295-plus,0.045-plus,0);
  ws_points[1]=Point3d(0.9+plus,0.045-plus,0);
  ws_points[2]=Point3d(0.9+plus,0.55+plus,0);
  ws_points[3]=Point3d(0.295-plus,0.55+plus,0);
  
  vector<Point2d> board_proj_pts(4), ws_proj_pts(4);

  Mat K=K_.clone();
  K.at<double>(0,2)*=10;
  K.at<double>(1,2)*=10;
  cv_ext::PinholeCameraModel camera_model( K, im_size.width*10000, im_size.height*10000, cv::Mat() );
  cv_ext::PinholeSceneProjector projector(camera_model);
  Eigen::Affine3d pose(camera_pose);
  Eigen::Quaterniond q(pose.linear());
  projector.setTransformation(q, pose.translation());
  projector.projectPoints(board_points, board_proj_pts);
  projector.projectPoints(ws_points, ws_proj_pts);
  
  Point board_v[4], ws_v[4];
  for(int i = 0; i < 4; ++i)
  {
    board_v[i] = board_proj_pts[i];
    board_v[i].x -= 9*K_.at<double>(0,2);
    board_v[i].y -= 9*K_.at<double>(1,2);
    ws_v[i] = ws_proj_pts[i];
    ws_v[i].x -= 9*K_.at<double>(0,2);
    ws_v[i].y -= 9*K_.at<double>(1,2);
  }
  
  fillConvexPoly(board_mask, board_v, 4, Scalar(255));
  fillConvexPoly(ws_mask, ws_v, 4, Scalar(255));
}


bool SensorLocalizer::localizeBoard(const cv::Mat& im, Eigen::Affine3d ref_sys_transform, Eigen::Affine3d& board_pose,cv::Mat & detection_output, bool reload_board)
{
  bool res=false;
  vector< int > ids;
  vector< vector< Point2f > > corners, rejected;
  Vec3d rvec, tvec;

  cv::Mat tmp_img;

  im.copyTo(tmp_img);
//   cv::resize(tmp_img, tmp_img, cv::Size(), .5, .5);

  // detect markers
  aruco::detectMarkers(tmp_img, dictionary_, corners, ids, detectorParams_, rejected);
  aruco::refineDetectedMarkers(tmp_img, board_, corners, ids, rejected, K_, Dist_coeff_, 10.0f, 3.0f);

  // estimate board pose
  int markersOfBoardDetected = 0;
  if(ids.size() > 0)
    markersOfBoardDetected = aruco::estimatePoseBoard(corners, ids, board_, K_, Dist_coeff_, rvec, tvec);

  // draw results
  cv::Mat imageCopy;
  tmp_img.copyTo(imageCopy);

  // check for gray image -> convert to RGB for visualization
  if(imageCopy.channels() == 1)
    cvtColor(imageCopy, imageCopy, CV_GRAY2RGB);

  if(ids.size() > 0) {
    aruco::drawDetectedMarkers(imageCopy, corners, ids);
  }

  //   cout<<markersOfBoardDetected<<endl;
  if(markersOfBoardDetected > 3)
  {
    //OFFSET FROM BOARD
//     tvec[0]+=0.339-0.0067;
//     tvec[1]-=0.092-0.0067;

    Eigen::Vector3d ax(rvec[0], rvec[1], rvec[2]);
    Eigen::Vector3d t(tvec[0], tvec[1], tvec[2]);
    float ang=ax.norm();
    Eigen::AngleAxisd aa(ang, ax/ang);
    board_pose.translation()=t;
    board_pose.linear()=aa.toRotationMatrix();

    // print board pose
    //std::cout << "Board Pose:\n" << board_pose.matrix() << std::endl << std::endl;

    // move reference system
    Eigen::Affine3d board_pose_inverse = board_pose.inverse();
    board_pose_inverse = ref_sys_transform*board_pose_inverse;
    board_pose = board_pose_inverse.inverse();
    tvec[0] = board_pose.translation()(0);
    tvec[1] = board_pose.translation()(1);
    tvec[2] = board_pose.translation()(2);

    // draw axis
    aruco::drawAxis(imageCopy, K_, Dist_coeff_, rvec, tvec, .6);

    res=true;
  }

  // set detection output
  imageCopy.copyTo(detection_output);

  //imshow("detected_board", imageCopy);
  //waitKey(20);

  return res;
}

bool SensorLocalizer::localizeCheckerboard(const Mat& im, Eigen::Affine3d& board_pose)
{

}

