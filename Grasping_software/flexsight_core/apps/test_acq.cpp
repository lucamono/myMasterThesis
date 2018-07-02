#include <iostream>
#include <sstream>
#include<signal.h>
#include<unistd.h>

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
  camera->disableTrigger();
  camera->setAutoFrameRate();
  camera->setGain(5);
  printCameraInfo(*camera);
  camera->startAcquisition();

  double t_us = 30000.0;
  camera->setExposureTime( t_us );


  int counter = 0;
  while ( !exit_app )
  {
    auto imgs = camera->getNextImages();
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
      }
    }
    waitKey(1);
  }

//   char c;
//   while( true )
//   {
//     c = getchar();
//     if( c == ' ' )
//       break;
//
//     camera->softwareTrigger();
//     auto imgs = camera->getNextImages();
//     for( int i = 0; i < imgs.size(); i++ )
//     {
//       stringstream sstr;
//       sstr<<"Img ";
//       sstr<<i;
//       Mat resized_img;
//       cv::resize(imgs[i], resized_img, Size(), 0.5,0.5);
//       imshow(sstr.str(), resized_img);
//     }
//     waitKey(1);
//   }

  camera->stopAcquisition();
  camera->switchOffCameras();

  return 0;
}
