#include <iostream>
#include <sstream>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <boost/asio.hpp>

#include <opencv2/opencv.hpp>

#include "camera_driver.h"
#include "camera_utils.h"

#include "apps_const.h"

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

int main( int argc, char** argv )
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

  camera = new flexsight::CameraDriver();
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

  const char *serial_port_device = "/dev/ttyACM1";

  asio::io_service io;
  asio::serial_port port(io);

  port.open(serial_port_device);
  port.set_option(asio::serial_port_base::baud_rate(9600));

  usleep(1000000);

  char cmd;
  if( disable_laser )
  {
    cmd = 0x1;
    asio::write(port, asio::buffer(&cmd,1));
  }
  else
  {
    cmd = 0x0;
    asio::write(port, asio::buffer(&cmd,1));
  }


  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(14);
  printCameraInfo(*camera);
  camera->enableSoftwareTrigger();
  camera->startAcquisition();

  double t_us = 10000.0;
  camera->setExposureTime( t_us );


  uint32_t timeout = 200000;
  if( argc > 1 )
  {
    double freq = atof(argv[1]);
    timeout = uint32_t(1e6/freq);
  }

  uint32_t counter = 0;
  while ( !exit_app )
  {
    usleep(200000);
    camera->softwareTrigger();
    auto imgs = camera->getNextImages();
    for( int i = 0; i < imgs.size(); i++ )
    {
      if(!imgs[i].empty())
      {
        stringstream sstr;

        if( i == left_cam_idx )
          sstr<<"left_img";
        else
          sstr<<"right_img";

        Mat resized_img;
        cv::resize(imgs[i], resized_img, Size(), 0.4,0.4);
        imshow(sstr.str(), resized_img);
//         imshow(sstr.str(), imgs[i]);
        waitKey(1);

        sstr<<"_";
        sstr<<std::setfill('0') << std::setw(6)<<counter;
        sstr<<".png";
        string file_path = "images/";
        file_path += sstr.str();
        imwrite(file_path, imgs[i]);
      }
    }
    counter++;
  }

  camera->stopAcquisition();
  camera->disableTrigger();
  camera->switchOffCameras();

  delete camera;

  return 0;
}
