#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>


#include <opencv2/opencv.hpp>
#include <cv_ext/cv_ext.h>

#include <eigen3/Eigen/Dense>

#include "camera_driver.h"
#include "camera_utils.h"
#include "camera_calibration.h"
#include "serial_communication.h"

#include "HalconCpp.h"
#include "HDevEngineCpp.h"

// #include <boost/asio.hpp>

using namespace std;
using namespace cv;
using namespace serial_communication;


flexsight::CameraDriver *camera;
cv_ext::MertensHDR<uchar> mertens_hdr;
double init_t_us;

static int fd;
static int fd_arduino;



// char mask_on = 0x6, mask_off = 0x1;
char mask_on = 0x7, mask_off = 0x1, laser_on = 0x0, laser_off = 0x1;
// char mask_on = 0x7, mask_off = 0x1, laser_on = 0x0, laser_off = 0x1;

//MODEL STUFF
std::string obj_sm3_path_, obj_sm3_flipped_path_;
std::string model_cad_path_;
std::string obj_cad_path_;
//HALCON STUFF
HalconCpp::HShapeModel3D ShapeModel3DID_l_, ShapeModel3DID_flipped_l_, ShapeModel3DID_r_, ShapeModel3DID_flipped_r_;
HalconCpp::HTuple CamParams_;
HalconCpp::HWindow *window_;
bool window_created_=false;
double obj_score_threshold_;
float im_scale_=1/2.0;

Mat left_mask_0, left_mask_1, right_mask_0, right_mask_1;
Mat ref_img;

cv_ext::PinholeCameraModel cm_left, cm_right;
Eigen::Isometry3d T_hand2eye, T_hand2eye_l, T_hand2eye_r;

void create3DShapeModel()
{
  
  float focal_lenght, sx_l, sy_l, cx_l, cy_l, im_width, im_height, sx_r, sy_r, cx_r, cy_r;
//   im_width = 2048*im_scale_;
//   im_height = 1536*im_scale_;
//   cx_l = 1086.2690664452496*im_scale_;
//   cy_l = 762.43053406785577*im_scale_;
//   focal_lenght = 0.01*im_scale_;
  im_width = (float)cm_left.imgWidth()*im_scale_;
  im_height = (float)cm_left.imgHeight()*im_scale_;
  cx_l = (float)cm_left.cx()*im_scale_;
  cy_l = (float)cm_left.cy()*im_scale_;
  focal_lenght = 0.01*im_scale_;
  sx_l=0.01/(float)cm_left.fx();
  sy_l=0.01/(float)cm_left.fy();
  
  obj_cad_path_="/home/marco/FlexSight/flexsight_core/bin/Lastra.stl";
  obj_sm3_path_="/home/marco/FlexSight/flexsight_core/models/Lastra.sm3";
  obj_sm3_flipped_path_="/home/marco/FlexSight/flexsight_core/models/Lastra_flipped.sm3";

  HalconCpp::HTuple camParam_l = HalconCpp::HTuple(focal_lenght).
      Append(HalconCpp::HTuple(0.0)).
      Append(HalconCpp::HTuple(sx_l)).
      Append(HalconCpp::HTuple(sy_l)).
      Append(HalconCpp::HTuple(cx_l)).
      Append(HalconCpp::HTuple(cy_l)).
      Append(HalconCpp::HTuple(im_width)).
      Append(HalconCpp::HTuple(im_height));
  HalconCpp::HTuple objectModel3DIDTuple;
  HalconCpp::HTuple status;
  HalconCpp::HTuple tempShapeModel3DID, tempShapeModel3DID_flipped;

  HalconCpp::ReadObjectModel3d(HalconCpp::HString(obj_cad_path_.c_str()),
                                HalconCpp::HString("mm"),
                                HalconCpp::HTuple(),
                                HalconCpp::HTuple(),
                                &objectModel3DIDTuple,
                                &status);
  HalconCpp::HObjectModel3D objectModel3DID_l;
  objectModel3DID_l.SetHandle(objectModel3DIDTuple);
  HalconCpp::PrepareObjectModel3d(objectModel3DID_l,
                                  HalconCpp::HString("shape_based_matching_3d"),
                                  HalconCpp::HString("true"),
                                  HalconCpp::HTuple(),
                                  HalconCpp::HTuple());
  HalconCpp::CreateShapeModel3d(objectModel3DID_l,
                                camParam_l,
                                HalconCpp::HTuple(0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HString("gba"),
                                HalconCpp::HTuple(-15 * M_PI/ 180.0),
                                HalconCpp::HTuple(15 * M_PI / 180.0),
                                HalconCpp::HTuple(-15.0 * M_PI / 180.0),
                                HalconCpp::HTuple(15.0 * M_PI / 180.0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HTuple(360 * M_PI / 180.0),
                                HalconCpp::HTuple(1.34),
                                HalconCpp::HTuple(1.46),
                                HalconCpp::HTuple(10),
                                HalconCpp::HTuple(),
                                HalconCpp::HTuple(),
                                &tempShapeModel3DID);
  ShapeModel3DID_l_.SetHandle(tempShapeModel3DID);
  //std::cout << ShapeModel3DID_.GetShapeModel3dParams("cam_param").ToString() << std::endl;
  ShapeModel3DID_l_.WriteShapeModel3d(obj_sm3_path_.c_str());
  //HalconCpp::WriteShapeModel3d(*tempShapeModel3DID, HalconCpp::HString(obj_sm3_full_path.c_str()));
  
  HalconCpp::CreateShapeModel3d(objectModel3DID_l,
                                camParam_l,
                                HalconCpp::HTuple(-180.0 * M_PI/ 180.0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HString("gba"),
                                HalconCpp::HTuple(-15 * M_PI/ 180.0),
                                HalconCpp::HTuple(15 * M_PI / 180.0),
                                HalconCpp::HTuple(-15.0 * M_PI / 180.0),
                                HalconCpp::HTuple(15.0 * M_PI / 180.0),
                                HalconCpp::HTuple(0),
                                HalconCpp::HTuple(360 * M_PI / 180.0),
                                HalconCpp::HTuple(1.34),
                                HalconCpp::HTuple(1.46),
                                HalconCpp::HTuple(10),
                                HalconCpp::HTuple(),
                                HalconCpp::HTuple(),
                                &tempShapeModel3DID_flipped);
  ShapeModel3DID_flipped_l_.SetHandle(tempShapeModel3DID_flipped);
  //std::cout << ShapeModel3DID_.GetShapeModel3dParams("cam_param").ToString() << std::endl;
  ShapeModel3DID_flipped_l_.WriteShapeModel3d(obj_sm3_flipped_path_.c_str());
}

void initHalcon()
{
  left_mask_0=imread("/home/marco/FlexSight/flexsight_core/bin/left_mask_0.png", CV_LOAD_IMAGE_UNCHANGED);
  left_mask_1=imread("/home/marco/FlexSight/flexsight_core/bin/left_mask_1.png", CV_LOAD_IMAGE_UNCHANGED);
  right_mask_0=imread("/home/marco/FlexSight/flexsight_core/bin/right_mask_0.png", CV_LOAD_IMAGE_UNCHANGED);
  right_mask_1=imread("/home/marco/FlexSight/flexsight_core/bin/right_mask_1.png", CV_LOAD_IMAGE_UNCHANGED);
  cv::resize(left_mask_0, left_mask_0, cv::Size(), im_scale_, im_scale_);
  cv::resize(left_mask_1, left_mask_1, cv::Size(), im_scale_, im_scale_);
  cv::resize(right_mask_0, right_mask_0, cv::Size(), im_scale_, im_scale_);
  cv::resize(right_mask_1, right_mask_1, cv::Size(), im_scale_, im_scale_);
  
  create3DShapeModel();
  ShapeModel3DID_l_.ReadShapeModel3d(obj_sm3_path_.c_str());
  ShapeModel3DID_flipped_l_.ReadShapeModel3d(obj_sm3_flipped_path_.c_str());
}

void clearProc(void* ptr) {
  return;
}

Eigen::Isometry3d processImageHalcon(Mat& img, Mat& mask, Eigen::Isometry3d& T_heye, double& score, bool& flag_flipped)
{
  HalconCpp::HImage h_img, h_mask, h_reduced_img;
  GenImage1Extern(&h_img,
                  "byte",
                  img.cols, img.rows, (long)img.data,
                  (long)clearProc);
  GenImage1Extern(&h_mask,
                  "byte",
                  mask.cols, mask.rows, (long)mask.data,
                  (long)clearProc);
  HalconCpp::HRegion mask_region = h_mask >= 200;
  h_reduced_img=h_img & mask_region;
  if (!window_created_) {
    window_ =  new HalconCpp::HWindow(0, 0, img.cols, img.rows);
    //std::cout << "dimensions: " << img_cv.cols << "," << img_cv.rows << std::endl;
    window_created_ = true;
  }

  
  
  window_->DispObj(h_img);

  //FIND THE OBJECT
  HalconCpp::HTuple ObjPose, ObjCovPose, ObjScore;
  HalconCpp::HTuple ObjPose_fl, ObjCovPose_fl, ObjScore_fl, ObjScore_tot, ObjPose_tot;
  HalconCpp::FindShapeModel3d(h_reduced_img,
                              ShapeModel3DID_l_,
                              0.7,
                              0.9,
                              3,
                              HalconCpp::HTuple("num_matches").Append(HalconCpp::HTuple("pose_refinement")),
                              HalconCpp::HTuple(5).Append(HalconCpp::HTuple("least_squares_very_high")),
                              &ObjPose,
                              &ObjCovPose,
                              &ObjScore);
  HalconCpp::FindShapeModel3d(h_reduced_img,
                              ShapeModel3DID_flipped_l_,
                              0.7,
                              0.9,
                              3,
                              HalconCpp::HTuple("num_matches").Append(HalconCpp::HTuple("pose_refinement")),
                              HalconCpp::HTuple(5).Append(HalconCpp::HTuple("least_squares_very_high")),
                              &ObjPose_fl,
                              &ObjCovPose_fl,
                              &ObjScore_fl);
  
  
  double max_score=0;
  ObjScore_tot=ObjScore.Clone();
  ObjScore_tot.Append(ObjScore_fl);
  for(int i=0; i<ObjScore_tot.Length(); i++)
  {
    if(max_score<ObjScore_tot[i])
      max_score=ObjScore_tot[i];
  }
  
  score=max_score;
  
  Eigen::Isometry3d obj_pose;
  flag_flipped=false;
  
  //DRAW POSES

  for (int i=0; i<ObjScore.Length(); ++i) {
//     std::cout << "Obj Score: " << ObjScore.ToString() << std::endl;
//     if (ObjScore[i] >= obj_score_threshold_) {
      HalconCpp::HObject ObjModelContour;
      double PoseI[7] = {ObjPose[i * 7],
                          ObjPose[i * 7 + 1],
                          ObjPose[i * 7 + 2],
                          ObjPose[i * 7 + 3],
                          ObjPose[i * 7 + 4],
                          ObjPose[i * 7 + 5]};
      HalconCpp::HTuple HPoseI(PoseI, 7);
      HalconCpp::ProjectShapeModel3d(&ObjModelContour,
                                      ShapeModel3DID_l_,
                                      ShapeModel3DID_l_.GetShapeModel3dParams("cam_param"),
                                      HPoseI,
                                      HalconCpp::HString("true"),
                                      0.523599);
      if(ObjScore[i]==max_score)
      {
        window_->SetColor("red");
        HalconCpp::HTuple h_q; 
        HalconCpp::PoseToQuat(ObjPose, &h_q);
        
        double q_I[4] = {h_q[i * 4],
                          h_q[i * 4 + 1],
                          h_q[i * 4 + 2],
                          h_q[i * 4 + 3]};
        
//         cout<<h_q.ToString()<<endl;
        obj_pose.translation()=Eigen::Vector3d(PoseI[0], PoseI[1], PoseI[2]);
        obj_pose.linear()=Eigen::Quaterniond(q_I[0], q_I[1], q_I[2], q_I[3]).toRotationMatrix();
      }
      else
        window_->SetColor("green");
      window_->DispObj(ObjModelContour);
//     }
  } 
  //DRAW POSES flipped
  for (int i=0; i<ObjScore_fl.Length(); ++i) {
    //std::cout << "Obj Score: " << ObjScore.ToString() << std::endl;
//     if (ObjScore[i] >= obj_score_threshold_) {
      HalconCpp::HObject ObjModelContour;
      double PoseI[7] = {ObjPose_fl[i * 7],
                          ObjPose_fl[i * 7 + 1],
                          ObjPose_fl[i * 7 + 2],
                          ObjPose_fl[i * 7 + 3],
                          ObjPose_fl[i * 7 + 4],
                          ObjPose_fl[i * 7 + 5]};
      HalconCpp::HTuple HPoseI(PoseI, 7);
      HalconCpp::ProjectShapeModel3d(&ObjModelContour,
                                      ShapeModel3DID_flipped_l_,
                                      ShapeModel3DID_flipped_l_.GetShapeModel3dParams("cam_param"),
                                      HPoseI,
                                      HalconCpp::HString("true"),
                                      0.523599);
      if(ObjScore_fl[i]==max_score)
      {
        flag_flipped=true;
        window_->SetColor("red");
        HalconCpp::HTuple h_q; 
        HalconCpp::PoseToQuat(ObjPose_fl, &h_q);
        
        double q_I[4] = {h_q[i * 4],
                          h_q[i * 4 + 1],
                          h_q[i * 4 + 2],
                          h_q[i * 4 + 3]};
        
//         cout<<h_q.ToString()<<endl;
        obj_pose.translation()=Eigen::Vector3d(PoseI[0], PoseI[1], PoseI[2]);
        obj_pose.linear()=Eigen::Quaterniond(q_I[0], q_I[1], q_I[2], q_I[3]).toRotationMatrix();
//         obj_pose.linear()=Eigen::Quaterniond(0, 0, 0, 1).toRotationMatrix();
//         cout<<endl<<obj_pose.matrix()<<endl<<endl;
      }
      else
        window_->SetColor("green");
      window_->DispObj(ObjModelContour);
//     }
  } 
  
  Eigen::Isometry3d dummy;
  dummy.translation()(2)=1000000;
  if(obj_pose.translation()(2)>0)
  {
    return T_heye.inverse()*obj_pose;
//      return obj_pose*T_heye.inverse();
  }
  else
    return dummy;
}

void ouch(int sig)
{
  printf("\n\nOUCH! − I got signal %d\n", sig);
  printf("Time to exit\n\n");
  
  if(camera)
  {
    camera->stopAcquisition();
    camera->disableTrigger();
    camera->switchOffCameras();
  }
  
  destroyAllWindows();
  
  close (fd);
  usleep(100000);
  close (fd_arduino);
  printf("Bye bye!\n\n");

  exit(EXIT_SUCCESS);
}

bool init_sensor(const char *serial_port, bool disable_laser )
{
  if( disable_laser )
    mask_on = 0x7;

//   const char *serial_port = "/dev/ttyACM0";

  if( (fd_arduino = openSerialPort( serial_port, BAUD_RATE_9600, PS_8N1 )) < 0 )
  {
    cout<<"can't open device "<<serial_port<<endl;
    return false;
  }

  camera = new flexsight::CameraDriver();
  camera->searchCameras();
  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(20);
  printCameraInfo(*camera);
  camera->enableHardwareTrigger();
  camera->startAcquisition();

  init_t_us = 400.0;
  double t_us = init_t_us;
  camera->setExposureTime( t_us );

  usleep(1000000);
  send( fd_arduino, &mask_off, 1 );
  
  cout<<"flexsight sensor initialized"<<endl;
  
//   usleep(10000);
//   send( fd_arduino, &mask_on, 1 );
//   auto imgs = camera->getNextImages();
//   send( fd_arduino, &mask_off, 1 );
  
  mertens_hdr.enableParallelism(true);
  mertens_hdr.setContrastExponent(1);
  mertens_hdr.setSaturationExponent(1);
  mertens_hdr.setExposednessExponent(1);
  
  return true;
}

void close_sensor()
{
  
}

void getImages(Mat& left, Mat& right)
{
  send( fd_arduino, &mask_on, 1 );
  auto imgs = camera->getNextImages();
  send( fd_arduino, &mask_off, 1 );
  
  left=imgs[1].clone();
  right=imgs[0].clone();
}

void getHdrImages(Mat& left, Mat& right)
{
  vector<Mat> imgs_seq[2];
  
  double t_us = init_t_us;
  for( int i = 0; i < 2; i++ )
    imgs_seq[i].clear();
  /*asio::write(port, asio::buffer(&laser_on,1))*/;
  
  while( t_us <= 5000 )
  {
//     cout<<t_us<<endl;
    camera->setExposureTime( t_us );
    send( fd_arduino, &mask_on, 1 );
    auto imgs = camera->getNextImages();
    send( fd_arduino, &mask_off, 1 );
    for( int i = 0; i < imgs.size(); i++ )
    {
      if(!imgs[i].empty())
        imgs_seq[i].push_back(imgs[i]);
    }
    t_us *= 2;
    send( fd_arduino, &laser_on, 1 );
    send( fd_arduino, &laser_off, 1 );
  }
  
  for( int i = 0; i < 2; i++ )
  {
    Mat hdr_img, resized_hdr_img, uchar_hdr_img;
    cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens(1,1,1);
    merge_mertens->process(imgs_seq[i], hdr_img);
    cv::normalize(hdr_img,uchar_hdr_img, 0, 255, cv::NORM_MINMAX, cv::DataType<uchar>::type);
    if(i==1)
      left=uchar_hdr_img.clone();
    else
      right=uchar_hdr_img.clone();
  }
}

void showImages(Mat& left, Mat& right)
{
  Mat resized_left, resized_right;
  cv::resize(left, resized_left, Size(), 0.5,0.5);
  imshow("left", resized_left);
  cv::resize(right, resized_right, Size(), 0.5,0.5);
  imshow("right", resized_right);
  waitKey(10);
}

void demo_example()
{
  srand(time(0));
  int iter=0;
  bool state_A=true;
  int iterations=0;
  while(1)
  {
    while(readHandShaking(fd)!=0);
    
    Mat im_left, im_right, im_left_resized, im_right_resized;
//     getImages(im_left, im_right);
    getHdrImages(im_left, im_right);
    
//     showImages(im_left, im_right);
    
    cv::resize(im_left, im_left_resized, cv::Size(), im_scale_, im_scale_);
    cv::resize(im_right, im_right_resized, cv::Size(), im_scale_, im_scale_);
    double score_l, score_r;
    Eigen::Isometry3d obj_pose, obj_pose_l, obj_pose_r;
    bool flipped, flipped_l, flipped_r;
    if(state_A)
    {
      obj_pose_l=processImageHalcon(im_left_resized, left_mask_0, T_hand2eye_l, score_l, flipped_l);
      obj_pose_r=processImageHalcon(im_right_resized, right_mask_0, T_hand2eye_r, score_r, flipped_r);
    }
    else
    {
      obj_pose_l=processImageHalcon(im_left_resized, left_mask_1, T_hand2eye_l, score_l, flipped_l);
      obj_pose_r=processImageHalcon(im_right_resized, right_mask_1, T_hand2eye_r, score_r, flipped_r);
    }
    if(score_l>=score_r)
    {
      T_hand2eye=Eigen::Isometry3d(T_hand2eye_l);
      obj_pose=Eigen::Isometry3d(obj_pose_l);
      ref_img=im_left.clone();
      flipped=flipped_l;
    }
    else
    {
      T_hand2eye=Eigen::Isometry3d(T_hand2eye_r);
      obj_pose=Eigen::Isometry3d(obj_pose_r);
      ref_img=im_right.clone();
      flipped=flipped_r;
    }
    
    cout<<obj_pose.translation().transpose()<<endl;
    


    double pos[6];
    
    if(obj_pose.translation()(2)<.08)
    {
      Eigen::Vector3d rpy=obj_pose.linear().eulerAngles(2,1,0);
      rpy(0)*=180.0f/M_PI;
      rpy(1)*=180.0f/M_PI;
      rpy(2)*=180.0f/M_PI;
      if(rpy(1)>90)
        rpy(1)-=180;
      else if(rpy(1)<-90)
        rpy(1)+=180;
      
      if(rpy(2)>90)
        rpy(2)-=180;
      else if(rpy(2)<-90)
        rpy(2)+=180;
      
      cout<<endl<<"rpy:\n"<<rpy.transpose()<<endl;
      
      if(rpy(1)>6)
        rpy(1)=6;
      else if(rpy(1)<-6)
        rpy(1)=-6;
      
      if(rpy(2)>6)
        rpy(2)=6;
      else if(rpy(2)<-6)
        rpy(2)-6;
        
      if(flipped)
        rpy(0)-=180;
      
//       if(iterations%2==0)
//         rpy(0)+=90;
//       
      iterations++;
      double rand_roll=(double)(rand()%180);
      
      cout<<endl<<"rpy:\n"<<rpy.transpose()<<endl;
      pos[0]=obj_pose.translation()(0)*1000.0f;
      pos[1]=obj_pose.translation()(1)*1000.0f;
      pos[2]=(obj_pose.translation()(2)*1000.0f);
//       pos[3]=rpy(0);//-90.0f;//-rpy(0);
//       pos[3]=rand_roll;
      if(!state_A)
        pos[3]=rand_roll;
      else
        pos[3]=rpy(0);//-90.0f;//-rpy(0);
      pos[4]=0;//rpy(1);//0;//-rpy(1)+M_PI;
      pos[5]=0;//-rpy(2);//0;//-rpy(2);
      
      /////////////////////AHHHHHHHHHHHHHHHHHHHHHH////////////////////////////////////
      if(!state_A)
      {
        pos[0]+=20; 
      }
      ////////////////////////////////////////////////////////////////////////
      
      sendPoseFrame(fd, pos);
      sendHandShaking(fd, 0);
      while(readHandShaking(fd)!=1)
        sendHandShaking(fd, 1);
      
      iter=0;
    }
    else
    {
      iter++; 
    }
    
    if (iter>5)
    {
      cout<<"EMPTY!"<<endl;
      pos[0]=0;
      pos[1]=1000;
      pos[2]=0;
      pos[3]=-90.0f;
      pos[4]=0;
      pos[5]=0;
      
      sendPoseFrame(fd, pos);
      sendHandShaking(fd, 0);
      while(readHandShaking(fd)!=1)
        sendHandShaking(fd, 1);
      
      state_A=!state_A;
//       return;
    }
//     else
//     {
//       srand(time (0));
// //       double pos[6];
//       double r=(double)(rand()%1000-500)/10.0f;
//       pos[0]=-12.0f+r;
//       r=(double)(rand()%1000-500)/10.0f;
//       pos[1]=151.0f+r;
//       pos[2]=20;
//       pos[3]=-90;
//       pos[4]=0;
//       pos[5]=0;
//     }
    
    
    Eigen::Isometry3d dbg_obj_pose(obj_pose);
    Eigen::Isometry3d dbg_obj_pose_l, dbg_obj_pose_r;
    dbg_obj_pose=T_hand2eye*dbg_obj_pose;
    dbg_obj_pose_l=T_hand2eye_l*T_hand2eye.inverse()*dbg_obj_pose;
    dbg_obj_pose_r=T_hand2eye_r*T_hand2eye.inverse()*dbg_obj_pose;
    Eigen::Vector3d obj_pos_l=dbg_obj_pose_l.translation();
    Eigen::Vector3d obj_pos_r=dbg_obj_pose_r.translation();
    
    cv_ext::PinholeSceneProjector sp(cm_left);
    Eigen::Quaterniond q; q.setIdentity();
    Eigen::Vector3d t(0,0,0);
    sp.setTransformation(q, t);
    
    Point3d p(dbg_obj_pose_l.translation()(0),dbg_obj_pose_l.translation()(1),dbg_obj_pose_l.translation()(2));
    vector<cv::Point3d> pts;
    pts.push_back(p);
    vector<cv::Point> im_pts;
    
    sp.projectPoints<cv::Point3d, cv::Point>(pts,im_pts);
    Mat dbg_im;
    cvtColor(im_left, dbg_im, CV_GRAY2BGR);
    cv_ext::drawCircles<cv::Point>(dbg_im,im_pts,10,cv::Scalar(0,0,255));
    resize(dbg_im,dbg_im,Size(),.5,.5);
    imshow("test_l",dbg_im);
    
    
    p=Point3d (dbg_obj_pose_r.translation()(0),dbg_obj_pose_r.translation()(1),dbg_obj_pose_r.translation()(2));
    pts.clear();
    pts.push_back(p);
    im_pts.clear();
    sp.projectPoints<cv::Point3d, cv::Point>(pts,im_pts);
    Mat dbg_im_r;
    cvtColor(im_right, dbg_im_r, CV_GRAY2BGR);
    cv_ext::drawCircles<cv::Point>(dbg_im_r,im_pts,10,cv::Scalar(0,0,255));
    resize(dbg_im_r,dbg_im_r,Size(),.5,.5);
    imshow("test_r",dbg_im_r);
    
    waitKey(10);


  }
}

void save_video()
{
  int iter=0;
  while(1)
  {
    
    Mat im_left, im_right;
    getImages(im_left, im_right);
    
    showImages(im_left, im_right);
    
    stringstream ss_left;
    ss_left<<"left_"<<iter<<".png";
    string name=ss_left.str();
    imwrite(name, im_left);
    stringstream ss_right;
    ss_right<<"right_"<<iter<<".png";
    name=ss_right.str();
    imwrite(name, im_right);
    
    iter++;
  }
}

void continous_detection()
{
  int iter=0;
  while(1)
  {
    
    Mat im_left, im_right, im_left_resized, im_right_resized;
//     getImages(im_left, im_right);
    getHdrImages(im_left, im_right);
    showImages(im_left, im_right);
        
    cv::resize(im_left, im_left_resized, cv::Size(), im_scale_, im_scale_);
    cv::resize(im_right, im_right_resized, cv::Size(), im_scale_, im_scale_);
    double score_l, score_r;
    bool flipped;
    Eigen::Isometry3d obj_pose;
    Eigen::Isometry3d obj_pose_l=processImageHalcon(im_left_resized, left_mask_0, T_hand2eye_l, score_l, flipped);
    Eigen::Isometry3d obj_pose_r=processImageHalcon(im_right_resized, right_mask_0, T_hand2eye_r, score_r, flipped);
    cout<<score_l<<" "<<score_r<<endl;
    if(score_l>=score_r)
    {
      T_hand2eye=T_hand2eye_l;
      obj_pose=obj_pose_l;
      ref_img=im_left.clone();
    }
    else
    {
      T_hand2eye=T_hand2eye_r;
      obj_pose=obj_pose_r;
      ref_img=im_right.clone();
    }
    
    cout<<obj_pose.translation().transpose()<<endl;
    cout<<endl<<"rpy:\n"<<obj_pose.linear().eulerAngles(2, 1, 0)<<endl;

    
    Eigen::Isometry3d dbg_obj_pose=T_hand2eye*obj_pose;
    Eigen::Vector3d obj_pos=dbg_obj_pose.translation();
    Point3d p(dbg_obj_pose.translation()(0),dbg_obj_pose.translation()(1),dbg_obj_pose.translation()(2));
    vector<cv::Point3d> pts;
    pts.push_back(p);
    vector<cv::Point> im_pts;
    cv_ext::PinholeSceneProjector sp(cm_left);
    Eigen::Quaterniond q; q.setIdentity();
    Eigen::Vector3d t(0,0,0);
    sp.setTransformation(q, t);
    sp.projectPoints<cv::Point3d, cv::Point>(pts,im_pts);
    Mat dbg_im;
    cvtColor(ref_img, dbg_im, CV_GRAY2BGR);
    cv_ext::drawCircles<cv::Point>(dbg_im,im_pts,10,cv::Scalar(0,0,255));
    cout<<im_pts[0]<<endl;
    resize(dbg_im,dbg_im,Size(),.5,.5);
    
    imshow("test",dbg_im);
    waitKey(10);
  }
}

void initCalibration()
{
  
  cv::Mat r_vec, t_vec;
  flexsight::readExtrinsicsFromFile( "/home/marco/FlexSight/flexsight_core/bin/left_handeye.yml", r_vec, t_vec );
  Eigen::Matrix4d T;
  cv_ext::exp2TransfMat(r_vec, t_vec, T);
  T_hand2eye_l.matrix()=T;
  
  cv::Mat r_vec2, t_vec2;
  Eigen::Matrix4d T2;
  flexsight::readExtrinsicsFromFile( "/home/marco/FlexSight/flexsight_core/bin/right_handeye.yml", r_vec2, t_vec2 );
  cv_ext::exp2TransfMat(r_vec2, t_vec2, T2);
  T_hand2eye_r.matrix()=T2;
  
  string left_camera_param_path="/home/marco/FlexSight/flexsight_core/bin/left_cam_model.yml";
  string right_camera_param_path="/home/marco/FlexSight/flexsight_core/bin/right_cam_model.yml";
  cm_left.readFromFile(left_camera_param_path);
  cm_right.readFromFile(right_camera_param_path);
  
}

int main(int argc, char* argv[])
{
  if(argc < 3)
    {
      fprintf(stderr,"Usage: %s <robot_device_name> <arduino_device_name> [disable_laser]", argv[0]);
      exit(EXIT_FAILURE);
    }


  if( (fd = openSerialPort( argv[1], BAUD_RATE_9600, PS_8E1 )) < 0 )
    {
      fprintf(stderr,"%s : can't open device %s\n!",argv[0], argv[1]);
//       exit(EXIT_FAILURE);
    }
    
  /* attivo l'handler per il segnale di CTLR-C */
  struct sigaction act;
  act.sa_handler = ouch;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  sigaction(SIGINT, &act, 0);
  
  printf("CTRL-C to exit\n\n");
  
  bool disable_laser=false;
  if(argc==4)
    if( strcmp(argv[3], "true") == 0 || strcmp(argv[3], "1") == 0)
      disable_laser=true;
    
  if(!init_sensor(argv[2], disable_laser))
    exit(EXIT_FAILURE);
  
  initCalibration();
  initHalcon();
  
  while(1) 
  {
    cout<<"type:\n  [1] for sending goal\n  [0] for hand shaking true\n  [9] for hand shaking false\n  [2] for reading hand shaking\n  [3] acquire images\n  [4] acquire HDR images\n  [6] save video frames\n  [7] continous_detection"<<endl;
    string input;
    Mat im_left, im_right;
    getline(cin, input);

    int option=0;
    option=stoi(input);
    switch (option)
    {
      int hs;
      case 1:
        cout<<"type [X Y Z A B C]"<<endl;
        string::size_type sz, sz_aux;
        getline(cin, input);
        double pos[6];
        pos[0]=stod (input,&sz);
        sz_aux=sz;
        pos[1]=stod (input.substr(sz),&sz);
        sz_aux+=sz;
        pos[2]=stod (input.substr(sz_aux),&sz);
        sz_aux+=sz;
        pos[3]=stod (input.substr(sz_aux),&sz);
        sz_aux+=sz;
        pos[4]=stod (input.substr(sz_aux),&sz);
        sz_aux+=sz;
        pos[5]=stod (input.substr(sz_aux),&sz);
        sz_aux+=sz;
        
        sendPoseFrame(fd, pos);
        break;
        
      case 2:
        hs=readHandShaking(fd);
        cout<<"hand shaking: "<<hs<<endl;
        break;
        
      case 0:
        sendHandShaking(fd, 1);
        break;
        
      case 9:
        sendHandShaking(fd, 0);
        break;
      case 5:
        demo_example();
        break;
      case 6:
        save_video();
        break;
      case 3:
        getImages(im_left, im_right);
        showImages(im_left, im_right);
        break;
      case 4:
        getHdrImages(im_left, im_right);
        showImages(im_left, im_right);
        break;
      case 7:
        continous_detection();
        break;
    }
  }
  
  exit(EXIT_SUCCESS);
}
