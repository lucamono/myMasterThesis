#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>


#include <opencv2/opencv.hpp>
#include <cv_ext/cv_ext.h>
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"


#include "camera_driver.h"
#include "camera_utils.h"

#include "apps_const.h"

class ObjectDetector
{

};


using namespace std;
using namespace boost;
namespace po = boost::program_options;


flexsight::CameraDriver *camera;

bool exit_app = false;

int fd;

void sig_handler(int signo)
{
  cout<<"Received signal "<<signo<<endl;
  exit_app = true;
}


using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  signal(SIGINT, sig_handler);
  signal(SIGKILL, sig_handler);

  string app_name( argv[0] ), model_filename, camera_filename, mask_filename, serial_port_device("/dev/ttyACM1");
  bool enable_laser = false;
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "mask_filename,i", po::value<string > ( &mask_filename )->required(),
    "Mask filename" )
  ( "serial_port_device,d", po::value<string > ( &serial_port_device ),
    "Serial port device" )
  ( "enable_laser,l", "Enable laser" )
  ( "right_camera,r", "Use right camera " );

  po::variables_map vm;
  bool has_color = false;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    if ( vm.count ( "enable_laser" ) )
      enable_laser = true;

    if ( vm.count ( "right_camera" ) )
      right_camera = true;

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  cout << "Loading model from file : "<<model_filename<< endl;
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;


  char mask_on = 0x7, mask_off = 0x1;

  if( enable_laser )
    mask_on = 0x6;


  asio::io_service io;
  asio::serial_port port(io);

  port.open(serial_port_device);
  port.set_option(asio::serial_port_base::baud_rate(9600));

  flexsight::CameraDriver();
  camera->searchCameras();

  auto cam_ids = camera->cameraIDs();
  int left_cam_idx, right_cam_idx;
  if( cam_ids.size() != 2 )
  {
    cout<<"Wrong number of camera(s). Exiting"<<endl;
    return -1;
  }
  else if( left_cam_id ==  stoi(cam_ids[0]) && right_cam_id ==  stoi(cam_ids[1]) )
  {
    left_cam_idx = 0;
    right_cam_idx = 1;
  }
  else if( left_cam_id ==  stoi(cam_ids[1]) && right_cam_id ==  stoi(cam_ids[0]) )
  {
    left_cam_idx = 1;
    right_cam_idx = 0;
  }
  else
  {
    cout<<"Camera(s) not found. Exiting"<<endl;
    return -1;
  }

  int camera_id = left_cam_idx;
  if( right_camera )
    camera_id = right_cam_idx;

  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(20);
  printCameraInfo(*camera);
  camera->enableHardwareTrigger();
  camera->startAcquisition();

  const double init_t_us = 400.0;
  double t_us = init_t_us;
  camera->setExposureTime( t_us );

  usleep(10000);


  Mat r_vec = (Mat_<double>(3,1) << 0,0,0),
      t_vec = (Mat_<double>(3,1) << 0,0,1.4);


  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);
  cam_model.setSizeScaleFactor(2);

  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setRenderWindowSize(cam_model.imgSize());
  obj_model.setCentroidOrigOffset();
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.setRenderFOV(M_PI/6);
  obj_model.enableVertexColors(has_color);

  if(!obj_model.setModelFile( model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();

  has_color = obj_model.vertexColorsEnabled();

  std::vector<cv::Point3d> cap_points;
  std::vector< Eigen::Quaterniond > quat_rotations;
  cv_ext::createIcospherePolarCap( cap_points, 2, M_PI/3/*, 1.0, true */);
//   cv_ext::show3DPoints(cap_points);
  cv_ext::sampleRotationsAroundVectors(cap_points, quat_rotations, 36 );

  cout<<"Num rotations :"<<quat_rotations.size()<<endl;

  for( double depth = 1.35; depth <= 1.45; depth+=0.02 )
  {
    for( int i = 0; i < quat_rotations.size(); i++ )
    {
      obj_model.setModelView(quat_rotations[i], Eigen::Vector3d(0.0,0.0,depth));
      obj_model.storeModelView();
    }
  }

  vector<Scalar> colors;
  colors.push_back(Scalar(0,0,255));
  colors.push_back(Scalar(255,0,0));
  colors.push_back(Scalar(0,255,0));
  colors.push_back(Scalar(255,0,255));

  for( int col = 255; col >= 0; col -= 5 )
    colors.push_back(Scalar(col,col,col));


  int increment = 4;

  DistanceTransform dc;
  Mat src_mask = imread ( mask_filename,cv::IMREAD_GRAYSCALE ), mask;
  cv::resize(src_mask, mask, cam_model.imgSize());

  cv_ext::showImage(mask);
  vector<Mat> imgs_seq[2];
  static int img_num = 0;
  while ( !exit_app )
  {

    t_us = init_t_us;
    for( int i = 0; i < 2; i++ )
      imgs_seq[i].clear();
    while( t_us <= 5000 )
    {
      camera->setExposureTime( t_us );
      asio::write(port, asio::buffer(&mask_on,1));
      auto imgs = camera->getNextImages();
      asio::write(port, asio::buffer(&mask_off,1));

      if(!imgs[camera_id].empty())
      {
        Mat resized_img;
        cv::resize(imgs[camera_id], resized_img, cam_model.imgSize());
        imgs_seq[camera_id].push_back(resized_img);
      }

      t_us *= 2;
    }


    Mat hdr_img, uchar_hdr_img;
    cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens(1,1,1);
    merge_mertens->process(imgs_seq[camera_id], hdr_img);

    stringstream sstr;
    sstr<<"Hdr ";
    sstr<<camera_id;
    cv::normalize(hdr_img,uchar_hdr_img, 0, 255, cv::NORM_MINMAX, cv::DataType<uchar>::type);
//     uchar_hdr_img &= mask;
//     imshow(sstr.str(), uchar_hdr_img);
//     waitKey(1);

//     sstr<<"_";
//     sstr<<img_num;
//     sstr<<".png";
//     imwrite(sstr.str(), uchar_hdr_img);
//     img_num++;


    Mat enanched_img;
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2);

    clahe->apply(uchar_hdr_img,enanched_img);

    LSDEdgeDetectorUniquePtr edge_detector_ptr( new LSDEdgeDetector() );
    edge_detector_ptr->setPyrNumLevels(2);
    edge_detector_ptr->setScale(2);
    edge_detector_ptr->setQuantizationThreshold(0.7);
    edge_detector_ptr->setMask(mask);
//     edge_detector_ptr->enableWhiteBackground(true);

//     CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   //   Best so far...
//     edge_detector_ptr->setLowThreshold(100);
//     edge_detector_ptr->setRatio(3);

    cv::Mat edge_map;
    edge_detector_ptr->setImage(enanched_img);
    edge_detector_ptr->getEdgeMap( edge_map );
    cv_ext::showImage(edge_map , "edge_map", true, 10);

    dc.enableParallelism(true);
    dc.setEdgeDetector(std::move(edge_detector_ptr));
    DirectionalChamferMatching dcm( cam_model );
    dcm.setTemplateModel(obj_model_ptr);

    cv_ext::BasicTimer timer;
    ImageTensorPtr dst_map_tensor_ptr;
    dc.computeDistanceMapTensor ( enanched_img, dst_map_tensor_ptr, 60, 6);
    dcm.setInput( dst_map_tensor_ptr );
    vector< TemplateMatch > matches;
    dcm.match(colors.size(), matches, (int)increment);

    cout<<"DCM elapsed time ms : "<<timer.elapsedTimeMs()<<endl;

    Mat dispaly;
    cv::cvtColor(uchar_hdr_img, dispaly,cv::COLOR_GRAY2BGR);
    for( int i_m = matches.size() - 1; i_m >= 0; i_m-- )
    {
      TemplateMatch &match = matches[i_m];
      obj_model.setModelView(match.r_quat, match.t_vec);
      vector<Point2f> proj_pts;
//       cout<<match.img_offset<<endl;
      obj_model.projectRasterPoints(proj_pts);
      for( int j = 0; j < proj_pts.size(); j++ )
      {
        proj_pts[j].x += match.img_offset.x;
        proj_pts[j].y += match.img_offset.y;
      }
//       cv::Mat display = img.clone();
//       cv_ext::drawPoints( display, proj_pts, Scalar(0,0,255) );
//       cv_ext::showImage(display,"display", true, 1000);

//         cv_ext::drawPoints( img, proj_pts, colors[i_m] );
        cv_ext::drawCircles( dispaly, proj_pts, 3, colors[i_m] );

    }
    cv_ext::showImage(dispaly,"display", true, 10);

  }


  camera->stopAcquisition();
  camera->disableTrigger();
  camera->switchOffCameras();

  delete camera;

  return 0;
}
