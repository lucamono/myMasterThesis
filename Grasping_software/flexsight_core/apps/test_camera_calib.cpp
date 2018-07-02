#include <iostream>
#include <sstream>
#include <string>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include "camera_calibration.h"
#include "file_tools.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char **argv)
{

  string app_name( argv[0] ), filelist_name, calibration_filename;
  bool show_corners = false, show_undistorted = false;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "filelist_name,f", po::value<string > ( &filelist_name )->required(),
    "Input images list file name" )
  ( "calibration_filename,c", po::value<string > ( &calibration_filename )->required(),
    "Output calibration basic filename" )
  ( "show_corners,s", "Show detected corners" )
  ( "show_undistorted,u", "Show undistorted images" );

  po::variables_map vm;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    if ( vm.count ( "show_corners" ) )
      show_corners = true;

    if ( vm.count ( "show_undistorted" ) )
      show_undistorted = true;

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

  cout << "Loading images filenames from file : "<<filelist_name<< endl;
  cout << "Saving camera parameters to file : "<<calibration_filename<< endl;

  std::vector<std::string> filelist;
  if( !readFileNames ( filelist_name, filelist ) )
    return -1;

  flexsight::CameraCalibration calib;
  calib.setBoardSize(Size(8,6));
  calib.setSquareSize(0.03983333333333333333);
  calib.setFileList(filelist);
  calib.extractImagePoints(2, show_corners);
  cout<<"RMS reprojection error :"<<calib.calibrate()<<endl;

  if (show_undistorted )
    calib.showUndistorted();

  auto cam_model = calib.getCamModel();
  cam_model.writeToFile(calibration_filename);
  
  return 0;
}