#include "Kinect2Interface.h"
#include <pcl/visualization/cloud_viewer.h>
#include "yoloDetector.h"
#include <cv_ext/cv_ext.h>
#include "mySensorLocalizer.h"
#include "serial_communication.h"


using namespace std;
using namespace serial_communication;

static int fd;
// executed with CTRL-C signal
void ouch(int sig)
{
  printf("\n\nOUCH! − I got signal %d\n", sig);
  printf("Time to exit\n\n");
  cv::destroyAllWindows();
  close (fd);
  printf("Bye bye!\n\n");
  exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{ 
  //init classes
  ImageProcessing im;
  Utils u;
  DatasetInterface d;
  yoloDetector darknet;
  MySensorLocalizer sl;
  
  //init communication usbToRobot
  if( (fd = openSerialPort( "/dev/ttyUSB0", BAUD_RATE_9600, PS_8E1 )) < 0 )
  {
      fprintf(stderr,": can't open device\n!");
      exit(EXIT_FAILURE);
  }
  
  /* attivo l'handler per il segnale di CTLR-C */
  struct sigaction act;
  act.sa_handler = ouch;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  sigaction(SIGINT, &act, 0);
  
  //######################################### INIT KINECT2 STEP #################################################################
  std::cout << "\033[1;32m---INIT KINECT2 DEVICE\033[0m" << std::endl; 
  processor freenectprocessor = OPENGL; 
  bool mirror_frame = true;
  Kinect2Interface k2_interface(freenectprocessor, mirror_frame);
  Kineckt2Frame acquisition;
  std::cout << "\033[1;32m[INIT KINECT2 DEVICE\033]:---DONE\033[0m" << std::endl; 
  //#############################################################################################################################
  
  
  
  //######################################### LOCALIZE BOARD FROM CAMERA STEP ###################################################
  std::cout << "\033[1;32m---LOCALIZE BOARD WRT CAMERA\033[0m" << std::endl; 
  //get acquisition frame
  k2_interface.getKinect2Frame(acquisition, false);
  cv::Mat image_for_board=acquisition.rgb_img;
  cvtColor(image_for_board, image_for_board, CV_BGRA2BGR);
  //localize board w.r.t camera
  Eigen::Affine3d board_pose = sl.localizeBoardFromCamera(image_for_board);
  std::cout << "\033[1;32m[LOCALIZE BOARD WRT CAMERA\033]:---DONE\033[0m" << std::endl; 
  //#############################################################################################################################
  
  
  
  //######################################## INIT OR LOAD DATASET ACTUAL STATE STEP #############################################
  std::cout << "\033[1;32m---INIT DATASET STATE DEVICE\033[0m" << std::endl; 
  //load or create new dataset
  int countScenario = d.initDataset();
  if(countScenario == -1) 
      return 0;
  //##############################################################################################################################
  
  
  
  //######################################### INIT THE NEURAL NETWORK SETUP STEP #################################################
  std::cout << "\033[1;32m---INIT YOLO NETWORK PARAMETERS\033[0m" << std::endl; 
  darknet.initDarknetParameters();
  std::cout << "\033[1;32m[INIT YOLO]:---DONE\033[0m" << std::endl; 
  //##############################################################################################################################
 
  Eigen::Vector3f eigenPoint3D;
  
  //for every grasping scenario
  for(int i = 0; i < 100; i++)
  {
      //init scenario
      std::cout << "\033[1;34m[SCENARIO " << countScenario << "]:---START...\033[0m" << std::endl; 
      int stepCounter =1;
      cvDestroyAllWindows();
      
      
      //######################################### KINECT2 ACQUISITION STEP ########################################################
      //start kinect2 acquisition for both rgb depth images
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---ACQUISITION KINECT2 DEVICE\033[0m" << std::endl; 
      k2_interface.getKinect2Frame(acquisition, false);
      cv::Mat image_rgb=acquisition.rgb_img;
      cv::Mat image_depth=acquisition.depth_img;
      //save image on file
      cv::imwrite("/home/luca/Scrivania/Grasping_software/config/temp/kinect_image.jpg",image_rgb);
      //load the current rgb image
      cv::Mat loaded_image = cv::imread("/home/luca/Scrivania/Grasping_software/config/temp/kinect_image.jpg", CV_LOAD_IMAGE_COLOR);
      //cv::imshow("ACQUIRED IMAGE_RGB",loaded_image);
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DONE\033[0m" << std::endl; 
      stepCounter++;
      //#############################################################################################################################
      

      
      //######################################### START YOLO DETECTOR STEP ##########################################################
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---START YOLO DETECTION\033[0m" << std::endl; 
      std::vector<boundingBox> bBoxes_norm = darknet.run_yolo_detector(loaded_image);    
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DONE\033[0m" << std::endl; 
      stepCounter++;
      //#############################################################################################################################
      
      
      
      //######################################### EVALUATE POINT OF GRASP WITH DEPTH SAFE STEP ######################################
      //take region of interest of rgb bounding box (loaded image and rgb image are used to correct channels' issues)   
      //take the depth z point by the mapping between registered kinect's images
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DEPTH VALIDATION\033[0m" << std::endl; 
      cv::Mat rgb_point_grasp_registered, rgb_registered, depth_point_grasp_registered, depth_registered, pointRandomGrasp;
      //until depth not valid
      bool notValid= true;
      std::pair<cv::Mat,cv::Point2f> pointAndMatrixGrasp;
      float real_depth;
      cv::Point2f xy_grasping_coords;
      cv::Point2f xy_registered_grasp_coords;
      while(notValid){
	  //take a random point of grasping inside the yolo's ROI 
	  pointAndMatrixGrasp = im.selectRandomPointGrasp(loaded_image, image_rgb, bBoxes_norm);
	  //check failure scenario
	  if(pointAndMatrixGrasp.second.x == -1000 && pointAndMatrixGrasp.second.y == -1000)
	      return 0;
	  //the rgb image with only the point of grasping
	  pointRandomGrasp = pointAndMatrixGrasp.first;
	  //coordinates of point of grasping
	  xy_grasping_coords = pointAndMatrixGrasp.second;
	  //performs registration for point of grasping
	  k2_interface.registerRGB(pointRandomGrasp,image_depth,rgb_point_grasp_registered,depth_point_grasp_registered);
	  //performs registration for image rgb-depth dataset
	  k2_interface.registerRGB(image_rgb,image_depth,rgb_registered,depth_registered);
	  //extract the depth parameter
	  std::pair <cv::Point2f,float> depthZ = im.getDepthPoint(rgb_point_grasp_registered, depth_point_grasp_registered);
	  //take registered image coordinates of point of grasping
	  xy_registered_grasp_coords = depthZ.first;
	  //check if the depth is good for grasping
	  if(im.checkDepthIsSafe(depth_point_grasp_registered, depthZ)){
	      notValid=false;
	      //depth offset (mm) to avoid collision with board
	      real_depth = depthZ.second;
	      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DONE\033[0m" << std::endl; 
	      stepCounter++;
	  }
      }
      //#############################################################################################################################
  
  
  
  
     //######################################### PERFORMS GRASPING STEP #############################################################
      //the random values of pitch and yaw rotations between -20 and 20 degrees
      bool grasped = true;
      float pitch,yaw;
      pitch = u.gen_random_float(-17, 17);
      yaw = u.gen_random_float(-17, 17);
  
      //convert 2D pixel Grasping point to 3D world grasping point
      eigenPoint3D = sl.pixelTo3DWorldPoint(xy_grasping_coords, real_depth,board_pose);
      
      
      cv::Mat debug_image2 = im.getDebugRgbImage(xy_grasping_coords);
      cv::imshow("POINT OF GRASPING",debug_image2);
      //cv::imwrite("/home/luca/Scrivania/debug.jpg",debug_image2);
      
      char key=cv::waitKey(0);
	switch(key)
	{
	    case 'n':
	    {
		grasped = false;
		break;
	    }
	    default:
	    break;
	}
        
      double posToPlanner[6];
      //set coords
      posToPlanner[0] = (double)eigenPoint3D(0);
      posToPlanner[1] = (double)eigenPoint3D(1);
      posToPlanner[2] = (double)eigenPoint3D(2) + 31.0;
      posToPlanner[3] = 90.0;
      if(pitch>=0) pitch = 180-pitch;
      else pitch = -180 - pitch;
      posToPlanner[4] = (double)pitch;
      posToPlanner[5] = (double)yaw;

      //403 241 81 -75 -17
      
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---GRASPING OBJECT\033[0m" << std::endl;    
      cv::Mat finish = cv::imread("/home/luca/Scrivania/Grasping_software/config/finish.jpg", CV_LOAD_IMAGE_COLOR);
      if(grasped)
      {
	  //send command to robot and start grasping
	  sendPoseFrame(fd, posToPlanner);
	  //sendHandShaking(fd, 0);
	  while(readHandShaking(fd)!=1)
	      sendHandShaking(fd, 1);
	  
	  
	  //wait robot grasping to end
	  while(readHandShaking(fd)!=0)
	  {
	    usleep(100000);
	  }
	  cv::imshow("GRASPING FINISH",finish);
	  char key=cv::waitKey(0);
	  switch(key)
	  {
	    case 'n':
	    {
		grasped = false;
		break;
	    }
	    case 'y':
	    {
		break;
	    }
	    default:
		grasped = false;
		break;
	}
	  
      }
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DONE\033[0m" << std::endl; 
      stepCounter++;
      //#############################################################################################################################
      
      
      
      
      //######################################### UPDATE DATASET STEP ###############################################################
      //update the dataset
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---SAVING DATASET...\033[0m" << std::endl;   
      //take the debug_image
      cv::Mat debug_image = im.getDebugRgbImage(xy_grasping_coords);
      //queue: rgbImageReg depthImageReg rgbNoRegImage depthNoRegImage world3Dpos_x world3Dpos_y world3Dpos_z posReg_x posReg_y posNoReg_x posNoReg_y depth pitch yaw graspingOnVacuumBoolean
      d.updateDataset(countScenario, rgb_registered, depth_registered, image_rgb, image_depth , debug_image, eigenPoint3D, xy_registered_grasp_coords, xy_grasping_coords, real_depth, pitch, yaw, grasped);
      std::cout << "\033[1;32m[STEP" << stepCounter << "]:---DONE\033[0m" << std::endl; 
      //#############################################################################################################################
      
      
      //the output after rgb-safedepth registration
      //cv::imshow("rgb_reg",rgb_registered);
      //cv::imshow("depth_reg",depth_registered);
      //cv::waitKey(0);
      countScenario++;
      //update the resumeState on dataset interface
      d.updateResumeState(countScenario);      
  }
  cvDestroyAllWindows();
  k2_interface.shutDown();
  return 0;
}

