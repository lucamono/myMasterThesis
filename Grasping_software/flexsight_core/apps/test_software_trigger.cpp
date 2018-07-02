#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>


#include <opencv2/opencv.hpp>

#include "camera_driver.h"
#include "camera_utils.h"

using namespace std;

flexsight::CameraDriver *camera;

bool exit_app = false;

void sig_handler(int signo)
{
  cout<<"Received signal "<<signo<<endl;
  exit_app = true;
}


using namespace std;
using namespace cv;

int main()
{
  signal(SIGINT, sig_handler);
  signal(SIGKILL, sig_handler);
  camera = new flexsight::CameraDriver();
  camera->searchCameras();
  camera->switchOnCameras();
  camera->setAutoFrameRate();
  camera->setGain(20);
  printCameraInfo(*camera);
  camera->enableSoftwareTrigger();
  camera->startAcquisition();

  double t_us = 10000.0;
  camera->setExposureTime( t_us );


  while ( !exit_app )
  {
    getchar();
    camera->softwareTrigger();
    auto imgs = camera->getNextImages();
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
//     camera->setExposureTime( t_us += 1000 );
  }


  camera->stopAcquisition();
  camera->disableTrigger();
  camera->switchOffCameras();

  return 0;
}
