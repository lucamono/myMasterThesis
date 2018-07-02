#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <boost/asio.hpp>

#include <opencv2/opencv.hpp>

#include "camera_driver.h"
#include "camera_utils.h"


using namespace std;
using namespace boost;

flexsight::CameraDriver *camera;

bool exit_app = false;

static int fd;

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

  char mask_on = 0x6, mask_off = 0x1;

  if( disable_laser )
    mask_on = 0x7;

  const char *serial_port_device = "/dev/ttyACM1";

  asio::io_service io;
  asio::serial_port port(io);

  port.open(serial_port_device);
  port.set_option(asio::serial_port_base::baud_rate(9600));

  camera = new flexsight::CameraDriver();
  camera->searchCameras();
  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(20);
  printCameraInfo(*camera);
  camera->enableHardwareTrigger();
  camera->startAcquisition();

  double t_us = 10000.0;
  camera->setExposureTime( t_us );

  usleep(10000);
  
  int counter = 0;
  while ( !exit_app )
  {
    asio::write(port, asio::buffer(&mask_on,1));
    auto imgs = camera->getNextImages();
    asio::write(port, asio::buffer(&mask_off,1));
    cout<<"Got "<<counter++<<" images"<<endl;
    for( int i = 0; i < imgs.size(); i++ )
    {
      if(!imgs[i].empty())
      {
        stringstream sstr;
        sstr<<"Img ";
        sstr<<i;
        Mat resized_img;
        cv::resize(imgs[i], resized_img, Size(), 0.5,0.5);
        imshow(sstr.str(), resized_img);
        waitKey(1);
      }
    }
  }

  camera->stopAcquisition();
  camera->disableTrigger();
  camera->switchOffCameras();

  return 0;
}
