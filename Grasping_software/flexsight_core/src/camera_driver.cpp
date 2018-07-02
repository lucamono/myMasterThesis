#include <iostream>
#include <sstream>
#include <functional>
#include <thread>

#include "cv_ext/timer.h"

#include "camera_driver.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

namespace flexsight
{

static std::function<void(vector<Mat> &)> imgCallack;

static bool win_created = false;
void defaultImgCallack( vector<Mat> &imgs )
{
  // TODO improve this
  for( int i = 0; i < imgs.size(); i++ )
  {
    stringstream sstr;
    sstr<<"Img ";
    sstr<<i;
    Mat resized_img;
    cv::resize(imgs[i], resized_img, Size(), 0.5,0.5);
    imshow(sstr.str(), resized_img);
  }
  waitKey(1);
}

CameraDriver::CameraDriver() :
  num_cams_(0),
  cam_acq_flag_(false),
  threaded_acq_(false),
  soft_trigger_enabled_(false),
  imgCallack_(defaultImgCallack)
{
  // Retrieve singleton reference to system object
  system_ptr_ = System::GetInstance();
}

CameraDriver::~CameraDriver()
{
}

void CameraDriver::searchCameras()
{
  try
  {
    // Retrieve list of cameras from the system
    CameraList cam_list = system_ptr_->GetCameras();
    num_cams_ = cam_list.GetSize();

    cam_ptrs_.clear();
    cam_ptrs_.reserve(num_cams_);
    cam_unique_ids_.clear();
    cam_unique_ids_.reserve(num_cams_);


    for (unsigned int i = 0; i < num_cams_; i++)
    {
      cam_ptrs_.push_back(cam_list.GetByIndex(i));
      gcstring unique_id = cam_ptrs_.back()->GetUniqueID();
      cam_unique_ids_.push_back(unique_id.c_str());
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setThreadCallback( std::function<void(std::vector<cv::Mat> &)> &callback )
{
  imgCallack_ = callback;
}

bool CameraDriver::startThreadedAcquisition()
{
  return startAcquisition( true );

}

bool CameraDriver::startAcquisition()
{
  return startAcquisition( false );

}

bool CameraDriver::startAcquisition( bool threaded )
{
  if( cam_acq_flag_ )
    return false;

  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      CEnumerationPtr acq_mode_ptr = cam_ptrs_[i]->GetNodeMap().GetNode("AcquisitionMode");
      if (!IsAvailable(acq_mode_ptr) || !IsWritable(acq_mode_ptr))
      {
        cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
        return false;
      }

      // Retrieve entry node from enumeration node
      CEnumEntryPtr acq_mode_continuous_ptr = acq_mode_ptr->GetEntryByName("Continuous");
      if (!IsAvailable(acq_mode_continuous_ptr) || !IsReadable(acq_mode_continuous_ptr))
      {
        cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
        return -1;
      }

      acq_mode_ptr->SetIntValue(acq_mode_continuous_ptr->GetValue());

      cam_ptrs_[i]->BeginAcquisition();
    }

    cam_acq_flag_ = true;

    if( threaded )
    {
      threaded_acq_ = true;
      cam_thread_ = std::thread( &CameraDriver::threadedAcquisition, this );
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }

  return true;
}

vector<Mat> CameraDriver::getNextImages()
{
  if( !cam_acq_flag_ || threaded_acq_ )
    return vector<Mat>();

  vector<ImagePtr> imgs(num_cams_);
  vector<Mat> cv_imgs(num_cams_);

  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
      imgs[i] = cam_ptrs_[i]->GetNextImage();

    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      if (imgs[i]->IsIncomplete())
      {
        cout << "Image "<<i<<" incomplete with image status " << imgs[i]->GetImageStatus() << "..." << endl << endl;
      }
      else
      {
        cv_imgs[i] = convert2OpenCV( imgs[i] );
      }
      imgs[i]->Release();
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
    return vector<Mat>();
  }
  return cv_imgs;
}

void CameraDriver::threadedAcquisition()
{
  try
  {
    vector<ImagePtr> imgs(num_cams_);
    vector<Mat> cv_imgs(num_cams_);

    while( cam_acq_flag_ )
    {
      for ( unsigned int i = 0; i < num_cams_; i++ )
        imgs[i] = cam_ptrs_[i]->GetNextImage();

      for ( unsigned int i = 0; i < num_cams_; i++ )
      {
        if (imgs[i]->IsIncomplete())
        {
          cout << "Image "<<i<<" incomplete with image status " << imgs[i]->GetImageStatus() << "..." << endl << endl;
        }
        else
        {
          cv_imgs[i] = convert2OpenCV( imgs[i] );
        }
        imgs[i]->Release();
      }
      imgCallack_(cv_imgs);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::stopAcquisition( )
{
  if( !cam_acq_flag_ )
    return;

  cam_acq_flag_ = false;

  if( threaded_acq_ )
    cam_thread_.join();

  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      cam_ptrs_[i]->EndAcquisition();
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::switchOnCameras()
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      cam_ptrs_[i]->Init();
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}


void CameraDriver::switchOffCameras()
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      if( cam_ptrs_[i]->IsInitialized() )
        cam_ptrs_[i]->DeInit();
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

double CameraDriver::getExposureTime()
{
  double exp_time = 0.0;
  if( num_cams_ )
  {
    try
    {
      exp_time = cam_ptrs_[0]->ExposureTime.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
  return exp_time;
}

double CameraDriver::getGain()
{
  double gain = 0.0;
  if( num_cams_ )
  {
    try
    {
      gain = cam_ptrs_[0]->Gain.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
  return gain;
}

double CameraDriver::getFrameRate()
{
  double fps = 0.0;
  if( num_cams_ )
  {
    try
    {
      fps = cam_ptrs_[0]->AcquisitionFrameRate.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
  return fps;
}

void CameraDriver::getExposureTimeLimits( double &min_t_us, double &max_t_us )
{
  min_t_us = max_t_us = 0;
  if( num_cams_ )
  {
    try
    {
      min_t_us = cam_ptrs_[0]->ExposureTime.GetMin();
      max_t_us = cam_ptrs_[0]->ExposureTime.GetMax();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
}

void CameraDriver::getGainLimits( double &min_gain, double &max_gain )
{
  min_gain = max_gain = 0;
  if( num_cams_ )
  {
    try
    {
      min_gain = cam_ptrs_[0]->Gain.GetMin();
      max_gain = cam_ptrs_[0]->Gain.GetMax();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
}

void CameraDriver::getFrameRateLimits( double &min_fps, double &max_fps )
{
  min_fps = max_fps = 0;
  if( num_cams_ )
  {
    try
    {
      min_fps = cam_ptrs_[0]->AcquisitionFrameRate.GetMin();
      max_fps = cam_ptrs_[0]->AcquisitionFrameRate.GetMax();
    }
    catch (Spinnaker::Exception &e)
    {
      cout << "Error: " << e.what() << endl;
    }
  }
}

void CameraDriver::setExposureTime( double t_us )
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      cam_ptrs_[i]->ExposureAuto.SetValue(ExposureAutoEnums::ExposureAuto_Off);
      cam_ptrs_[i]->ExposureMode.SetValue(ExposureModeEnums::ExposureMode_Timed);
      cam_ptrs_[i]->ExposureTime.SetValue(t_us);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setGain( double gain )
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      cam_ptrs_[i]->GainAuto.SetValue(GainAutoEnums::GainAuto_Off);
      cam_ptrs_[i]->Gain.SetValue(gain);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setFrameRate( double fps )
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      CBooleanPtr acq_fr_enable_ptr = cam_ptrs_[i]->GetNodeMap().GetNode("AcquisitionFrameRateEnable");
      if (!IsAvailable(acq_fr_enable_ptr) || !IsReadable(acq_fr_enable_ptr))
      {
        cout << "Unable to set the acquisition frame rate. Aborting..." << endl << endl;
        return;
      }

      acq_fr_enable_ptr->SetValue(true);
      cam_ptrs_[i]->AcquisitionFrameRate.SetValue(fps);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setAutoExposure()
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      cam_ptrs_[i]->ExposureAuto.SetValue(ExposureAutoEnums::ExposureAuto_Continuous);
      cam_ptrs_[i]->ExposureMode.SetValue(ExposureModeEnums::ExposureMode_Timed);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setAutoGain()
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
      cam_ptrs_[i]->GainAuto.SetValue(GainAutoEnums::GainAuto_Continuous);
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::setAutoFrameRate()
{
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      CBooleanPtr acq_fr_enable_ptr = cam_ptrs_[i]->GetNodeMap().GetNode("AcquisitionFrameRateEnable");
      if (!IsAvailable(acq_fr_enable_ptr) || !IsReadable(acq_fr_enable_ptr))
      {
        cout << "Unable to set the acquisition frame rate. Aborting..." << endl << endl;
        return;
      }

      acq_fr_enable_ptr->SetValue(false);
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

Mat CameraDriver::convert2OpenCV( ImagePtr &img )
{
  unsigned int x_padding = img->GetXPadding();
  unsigned int y_padding = img->GetYPadding();
  unsigned int rows = img->GetWidth();
  unsigned int cols = img->GetHeight();

  // image data contains padding. When allocating Mat container size,
  // you need to account for the X,Y image data padding.
  Mat cv_img = cv::Mat( cols + y_padding, rows + x_padding, cv::DataType< uchar >::type,
                        img->GetData(), img->GetStride());
  return cv_img;
}

bool CameraDriver::enableSoftwareTrigger()
{
  soft_trigger_enabled_ = true;
  cam_s_trigger_.clear();
  cam_s_trigger_.reserve(num_cams_);

  // TODO Check if it makes sense
  setAutoFrameRate();

  try
  {
    soft_trigger_enabled_ = true;
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      INodeMap &node_map = cam_ptrs_[i]->GetNodeMap();

      // Ensure trigger mode off
      CEnumerationPtr trigger_mode_ptr = node_map.GetNode("TriggerMode");
      if (!IsAvailable(trigger_mode_ptr) || !IsReadable(trigger_mode_ptr))
      {
        cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
        return false;
      }

      CEnumEntryPtr trigger_mode_off_ptr = trigger_mode_ptr->GetEntryByName("Off");
      if (!IsAvailable(trigger_mode_off_ptr) || !IsReadable(trigger_mode_off_ptr))
      {
        cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
        return false;
      }

      trigger_mode_ptr->SetIntValue(trigger_mode_off_ptr->GetValue());


      // Select trigger source
      CEnumerationPtr trigger_source_ptr = node_map.GetNode("TriggerSource");
      if (!IsAvailable(trigger_source_ptr) || !IsWritable(trigger_source_ptr))
      {
        cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
        return false;
      }


      CEnumEntryPtr trigger_software_ptr = trigger_source_ptr->GetEntryByName("Software");
      if (!IsAvailable(trigger_software_ptr) || !IsReadable(trigger_software_ptr))
      {
        cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
        return false;
      }

      trigger_source_ptr->SetIntValue(trigger_software_ptr->GetValue());


      // Turn trigger mode on
      CEnumEntryPtr trigger_mode_on_ptr = trigger_mode_ptr->GetEntryByName("On");
      if (!IsAvailable(trigger_mode_on_ptr) || !IsReadable(trigger_mode_on_ptr))
      {
        cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
        return false;
      }

      trigger_mode_ptr->SetIntValue(trigger_mode_on_ptr->GetValue());

      cam_s_trigger_.push_back(node_map.GetNode("TriggerSoftware"));
      if (!IsAvailable(cam_s_trigger_.back()) || !IsWritable(cam_s_trigger_.back()))
      {
        cout << "Unable to execute trigger. Aborting..." << endl;
        return false;
      }
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
    return false;
  }

  return true;
}

bool CameraDriver::enableHardwareTrigger()
{
  soft_trigger_enabled_ = false;

  // TODO Check if it makes sense
  setAutoFrameRate();

  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      INodeMap &node_map = cam_ptrs_[i]->GetNodeMap();

      // Ensure trigger mode off
      CEnumerationPtr trigger_mode_ptr = node_map.GetNode("TriggerMode");
      if (!IsAvailable(trigger_mode_ptr) || !IsReadable(trigger_mode_ptr))
      {
        cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
        return false;
      }

      CEnumEntryPtr trigger_mode_off_ptr = trigger_mode_ptr->GetEntryByName("Off");
      if (!IsAvailable(trigger_mode_off_ptr) || !IsReadable(trigger_mode_off_ptr))
      {
        cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
        return false;
      }

      trigger_mode_ptr->SetIntValue(trigger_mode_off_ptr->GetValue());

      // Select trigger source
      CEnumerationPtr trigger_source_ptr = node_map.GetNode("TriggerSource");
      if (!IsAvailable(trigger_source_ptr) || !IsWritable(trigger_source_ptr))
      {
        cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
        return false;
      }

      CEnumEntryPtr trigger_hardware_ptr = trigger_source_ptr->GetEntryByName("Line0");
      if (!IsAvailable(trigger_hardware_ptr) || !IsReadable(trigger_hardware_ptr))
      {
        cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
        return -1;
      }

      trigger_source_ptr->SetIntValue(trigger_hardware_ptr->GetValue());

      // Turn trigger mode on
      CEnumEntryPtr trigger_mode_on_ptr = trigger_mode_ptr->GetEntryByName("On");
      if (!IsAvailable(trigger_mode_on_ptr) || !IsReadable(trigger_mode_on_ptr))
      {
        cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
        return false;
      }

      trigger_mode_ptr->SetIntValue(trigger_mode_on_ptr->GetValue());
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
    return false;
  }

  return true;
}

void CameraDriver::disableTrigger()
{
  soft_trigger_enabled_ = false;
  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
    {
      INodeMap &node_map = cam_ptrs_[i]->GetNodeMap();

      // Ensure trigger mode off
      CEnumerationPtr trigger_mode_ptr = node_map.GetNode("TriggerMode");
      if (!IsAvailable(trigger_mode_ptr) || !IsReadable(trigger_mode_ptr))
      {
        cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
        return;
      }

      CEnumEntryPtr trigger_mode_off_ptr = trigger_mode_ptr->GetEntryByName("Off");
      if (!IsAvailable(trigger_mode_off_ptr) || !IsReadable(trigger_mode_off_ptr))
      {
        cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
        return;
      }

      trigger_mode_ptr->SetIntValue(trigger_mode_off_ptr->GetValue());
    }
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

void CameraDriver::softwareTrigger()
{
  if( !soft_trigger_enabled_ )
    return;

  try
  {
    for ( unsigned int i = 0; i < num_cams_; i++ )
      cam_s_trigger_[i]->Execute();
  }
  catch (Spinnaker::Exception &e)
  {
    cout << "Error: " << e.what() << endl;
  }
}

}