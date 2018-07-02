#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <boost/asio.hpp>

#include <opencv2/opencv.hpp>
#include <cv_ext/cv_ext.h>

#include "camera_driver.h"
#include "camera_utils.h"

using namespace std;
using namespace boost;

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

int main( int argc, char **argv )
{
  signal(SIGINT, sig_handler);
  signal(SIGKILL, sig_handler);

  bool disable_laser = false;
  if( argc > 1 )
  {

    for(int i = 1; i < argc; i++ )
    {
      if( strcmp(argv[i], "--laser-off") == 0 || strcmp(argv[i], "-lo") == 0 )
        disable_laser = true;
    }
  }

  char cam_on = 0x7, cam_off = 0x1, laser_on = 0x0, laser_off = 0x1;
/*
  if( disable_laser )
    mask_on = 0x7;*/

  const char *serial_port_device = "/dev/ttyACM1";

  asio::io_service io;
  asio::serial_port port(io);

  port.open(serial_port_device);
  port.set_option(asio::serial_port_base::baud_rate(9600));

  camera = new flexsight::CameraDriver();
  camera->searchCameras();
  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(14);
  printCameraInfo(*camera);
  camera->enableHardwareTrigger();
  camera->startAcquisition();

  const double init_t_us = 400.0;
  double t_us = init_t_us;
  camera->setExposureTime( t_us );

  usleep(10000);

  cv_ext::MertensHDR<uchar> mertens_hdr;
  mertens_hdr.enableParallelism(true);
  mertens_hdr.setContrastExponent(1);
  mertens_hdr.setSaturationExponent(1);
  mertens_hdr.setExposednessExponent(1);

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
      asio::write(port, asio::buffer(&cam_on,1));
      auto imgs = camera->getNextImages();
      asio::write(port, asio::buffer(&cam_off,1));

      for( int i = 0; i < imgs.size(); i++ )
      {
        if(!imgs[i].empty())
          imgs_seq[i].push_back(imgs[i]);
      }
      t_us *= 2;
    }

    asio::write(port, asio::buffer(&laser_on,1));

    for( int i = 0; i < 2; i++ )
    {
      Mat hdr_img, resized_hdr_img, uchar_hdr_img;
      cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens(1,1,1);
      merge_mertens->process(imgs_seq[i], hdr_img);

//         mertens_hdr.merge(imgs_seq[i], hdr_img);
      cv::resize(hdr_img, resized_hdr_img, Size(), 0.5,0.5);

      stringstream sstr;
      sstr<<"Hdr ";
      sstr<<i;
      imshow(sstr.str(), resized_hdr_img);
      waitKey(1);
//       sstr<<"_";
//       sstr<<img_num;
//       sstr<<".png";
//       cv::normalize(hdr_img,uchar_hdr_img, 0, 255, cv::NORM_MINMAX, cv::DataType<uchar>::type);
//
//       imwrite(sstr.str(), uchar_hdr_img);
    }
    asio::write(port, asio::buffer(&laser_off,1));
    img_num++;
    cout<<img_num<<endl;
//     if(!(img_num%10))
//     {
//       port.close();
//       asio::io_service io;
//       port = asio::serial_port(io);
//       port.open(serial_port_device);
//       port.set_option(asio::serial_port_base::baud_rate(9600));
//     }
  }


  camera->stopAcquisition();
  camera->disableTrigger();
  camera->switchOffCameras();

  port.close();

  return 0;
}
