#include <iostream>
#include "camera_utils.h"

using namespace std;

void printCameraInfo( flexsight::CameraDriver &cam )
{
  auto ids = cam.cameraIDs();
  for( int i = 0; i < cam.numCameras(); i++)
    cout<<"Cam "<<i<<" ID "<<ids[i]<<endl;

  double min, max;
  cam.getExposureTimeLimits( min, max );
  cout<<"Exposure time (uSec) : "<<cam.getExposureTime()<<" ["<<min<<", "<<max<<"]"<<endl;
  cam.getGainLimits( min, max );
  cout<<"Gain (dB) : "<<cam.getGain()<<" ["<<min<<", "<<max<<"]"<<endl;
  cam.getFrameRateLimits( min, max );
  cout<<"FrameRate (Hz) : "<<cam.getFrameRate()<<" ["<<min<<", "<<max<<"]"<<endl;
}