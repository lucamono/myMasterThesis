#include <iostream>
#include <sstream>
#include<signal.h>
#include<unistd.h>

#include <opencv2/opencv.hpp>

#include "camera_driver.h"
#include "camera_utils.h"

using namespace std;

flexsight::CameraDriver *camera;


void sig_handler(int signo)
{
  cout<<"Received signal "<<signo<<endl;
  camera->stopAcquisition();
  camera->switchOffCameras();
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
  printCameraInfo(*camera);
  camera->disableTrigger();
  camera->startThreadedAcquisition();


  double t_us = 30000.0;
  camera->setExposureTime( t_us );

  char c;
  while( c = getchar() )
  {
    getchar();
    printf("%c",c);
    if( c == '+' )
    {
      t_us += 10000;
      camera->setExposureTime( t_us );
    }
    else if( c == '-' )
    {
      t_us -= 10000;
      camera->setExposureTime( t_us );
    }
    else
      break;
  }

  camera->stopAcquisition();
  camera->switchOffCameras();

  return 0;
}
