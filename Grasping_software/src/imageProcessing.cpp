#include "imageProcessing.h"

cv::Mat ImageProcessing::applyMask(cv::Mat image){
    //create mask  
    cv::Mat mask = cv::imread("/home/luca/Scrivania/Grasping_software/config/mask.jpg", CV_LOAD_IMAGE_COLOR);
    cv::Mat masked(image.size(),CV_8UC3,cv::Scalar(255,255,255));
    image.copyTo(masked,mask);
    // cv::imshow("MASKED IMAGE",masked);
    return masked;
}

cv::Mat ImageProcessing::cropImage(cv::Mat image){
    //create boundary  
    cv::Rect myROI(550, 300, 550, 400);
    //crop the original image to the defined ROI 
    cv::Mat cropped_image = image(myROI);
    return cropped_image;
}

cv::Mat ImageProcessing::drawBbox(cv::Mat image, std::vector<boundingBox> bboxes){
    for(int i=0; i < bboxes.size(); ++i)
	cv::rectangle(image, bboxes.at(i).first, bboxes.at(i).second, cv::Scalar(0, 0, 255),2);
    return image;
}

cv::Mat ImageProcessing::filterBbox(cv::Mat image, std::vector<boundingBox> bboxes)
{
    cv::Mat result(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    int bboxesRow;
    int bboxesCols;
    for(int i=0; i < bboxes.size(); ++i)
    {
        bboxesRow =  bboxes.at(i).second.y - bboxes.at(i).first.y; 
        bboxesCols =  bboxes.at(i).second.x - bboxes.at(i).first.x;
	for(int rows=0; rows < bboxesRow; ++rows)
	{
	    for(int cols=0; cols < bboxesCols; ++cols)
	    {
		if((int)result.at<uchar>(cv::Point(bboxes.at(i).first.x + cols,  bboxes.at(i).first.y+rows))==0){
		      result.at<uchar>(cv::Point(bboxes.at(i).first.x + cols,  bboxes.at(i).first.y+rows)) = 255;
		}
	    }
	}
    }
    cv::imshow("FILTERED IMAGE_RGB",result);
    return result;
}

std::pair<cv::Mat,cv::Point2f> ImageProcessing::selectRandomPointGrasp(cv::Mat image, cv::Mat kinect_image, std::vector<boundingBox> bboxes)
{   
    cv::Mat initImage(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat bBoxesOnImage = initImage.clone();
    cv::Mat pointOnImage = initImage.clone();
    cv::Point2f resultPoint;
    //case of empty list of bounding boxes
    resultPoint.x = -1000;
    resultPoint.y = -1000;
    std::pair<cv::Mat,cv::Point2f> output;
    output.first = initImage;
    output.second = resultPoint;
    
    int RoiTop;
    int RoiLeft;
    int RoiBottom;
    int RoiRight;
    int bboxesRow;
    int bboxesCols;
    int rnd_BBox;
    int rnd_PixelX;
    int rnd_PixelY;
    if(bboxes.size()>0){

        std::srand(time(0));
	//prepare ROI bboxes to will choose random point of grasping
	RoiTop = bboxes.at(0).first.y;
	RoiLeft = bboxes.at(0).first.x;
	RoiBottom = bboxes.at(0).second.y;
	RoiRight = bboxes.at(0).second.x;
	bool maskYolo = false;
	for(int i=0; i < bboxes.size(); ++i)
	{   
	    //evaluate RoiBoundary pixels
	    if(bboxes.at(i).first.y < RoiTop){
		RoiTop = bboxes.at(i).first.y;
	    }
	    if(bboxes.at(i).first.x < RoiLeft){
		RoiLeft = bboxes.at(i).first.x;
	    }
	    if(bboxes.at(i).second.y > RoiBottom){
		RoiBottom = bboxes.at(i).second.y;
	    }
	    if(bboxes.at(i).second.x > RoiRight){
		RoiRight = bboxes.at(i).second.x;
	    }
	    //draw white pixels on the mask
	    //check good bboxes
	    if((abs(RoiLeft-RoiRight) < 300) || (abs(RoiTop-RoiBottom) < 300))
	    {
		maskYolo = true;
		bboxesRow =  bboxes.at(i).second.y - bboxes.at(i).first.y; 
		bboxesCols =  bboxes.at(i).second.x - bboxes.at(i).first.x;
		for(int rows=0; rows < bboxesRow; ++rows)
		{
		    for(int cols=0; cols < bboxesCols; ++cols)
		    {
			if((int)bBoxesOnImage.at<uchar>(cv::Point(bboxes.at(i).first.x + cols,  bboxes.at(i).first.y+rows))==0){
			      bBoxesOnImage.at<uchar>(cv::Point(bboxes.at(i).first.x + cols,  bboxes.at(i).first.y+rows)) = 255;
			}
		    }
		}
	    }      
	}
	
	int BBoxTop;
	int BBoxLeft;
	int BBoxBottom;
	int BBoxRight;
	if(maskYolo)
	{
	    int indexBBox = bboxes.size() - 1;
	    //the random choice is from at least 2 BBox
	    if(bboxes.size()>1)
	    {
		rnd_BBox = rand() % indexBBox;
	    }
	    else
	    {
		rnd_BBox = 0;
	    }	     
	    BBoxTop = bboxes.at(rnd_BBox).first.y;
	    BBoxLeft = bboxes.at(rnd_BBox).first.x;
	    BBoxBottom = bboxes.at(rnd_BBox).second.y;
	    BBoxRight = bboxes.at(rnd_BBox).second.x;
	    rnd_PixelX = rand() % (BBoxRight - BBoxLeft) + BBoxLeft;
	    rnd_PixelY = rand() % (BBoxBottom - BBoxTop) + BBoxTop;
	
	    //draw point on image as a 5x5 white grid
	    int gridRows = 5;
	    int gridCols = 5;
	    int offsetGrid = 2;
	    for(int rows=0; rows < gridRows; ++rows)
	    {
		for(int cols=0; cols < gridCols; ++cols)
		{
		    pointOnImage.at<uchar>(cv::Point(rnd_PixelX - offsetGrid + cols,  rnd_PixelY - offsetGrid + rows)) = 255;
		}
	    }
	    
	    //correct channel's issues about kinect image with loaded image
	    
	    //debug dataset image (region proposal + point of grasping)
	    kinect_image.copyTo(bBoxesOnImage, bBoxesOnImage);
	    //cv::imshow("BoundingBoxes on image",bBoxesOnImage);
	    //convert to gray grayScale
	    cvtColor( bBoxesOnImage, bBoxesOnImage, CV_BGR2GRAY );
	    cv::imwrite("/home/luca/Scrivania/Grasping_software/config/temp/debug_temp.jpg",bBoxesOnImage);
	    cv::Mat pointRandomImage;
	    kinect_image.copyTo(pointRandomImage, pointOnImage);
	    //cv::imshow("POINT OF GRASPING",pointOnImage);
	    output.first = pointRandomImage;
	    output.second = cv::Point(rnd_PixelX, rnd_PixelY);
	}
	else{
	    std::cout << "\033[1;31m[ERROR!]:---REGION PROPOSAL FAILURE IN THE SCENE- EXIT PROGRAM!\033[0m" << std::endl; 
	}
    }
    else{
	 std::cout << "\033[1;31m[ERROR!]:---NO OBJECT IN THE SCENE- EXIT PROGRAM!\033[0m" << std::endl; 
    }
    return output;
}

std::pair <cv::Point2f,float> ImageProcessing::getDepthPoint(cv::Mat rgb_registered, cv::Mat depth_registered){
      Utils u;
      std::pair <cv::Point2f,float> res;
      cv::Point2f temp;
      temp.x = -100;
      temp.y = -100;
      res.first =temp;
      res.second= -1.0;
      double min=0;
      double max=255;
      cv::Point min_loc, max_loc; 
      cv::Mat grayScale_registered;
      cv::cvtColor(rgb_registered, grayScale_registered, CV_BGR2GRAY);
      cv::minMaxLoc(grayScale_registered, &min, &max, &min_loc, &max_loc);
      cv::Mat depthMap = u.convert4ChannelsToFloatImage(depth_registered);
      res.first = max_loc;
      res.second = depthMap.at<float>(max_loc); 
      return res;
}

bool ImageProcessing::checkDepthIsSafe(cv::Mat depth_registered, std::pair <cv::Point2f,float> depthZ){
    bool goodDepth = false;
    //avoid to evaluate under 0,3 meter 
    int offset_detection = 300;
    //
    if(depthZ.second > offset_detection){
        
        //compute depth-average between nearest pixels (5x5 Grid)
	int gridRows = 5;
	int gridCols = 5;
	int offsetGrid = 2;
	//each depth pixels
	float temp_depth;
	float average = 0;
	//take in consideration the black pixels of the depth
	int blackDepth=0;
	//depth threshold between the average and the depthZ measured 
	float threshold_mm = 20;
	for(int rows=0; rows < gridRows; ++rows)
	{
	    for(int cols=0; cols < gridCols; ++cols)
	    {	
	        //case of more black depth pixels
	        if(blackDepth>5){
		     std::cout << "\033[1;33m[WARNING!]---NO SAFE DEPTH FOUNDED. RESTART STEP. \033[0m" << std::endl; 
		     return false;
		}
		//take the depth of the nearest pixel
		temp_depth = depth_registered.at<float>(cv::Point(depthZ.first.x - offsetGrid + cols,  depthZ.first.y - offsetGrid + rows));
		//case of black pixel
		if(temp_depth == 0){
		    blackDepth++;
		}
		average += temp_depth;
	    }
	}
	average -= depthZ.second;
	average = average/((gridRows*gridCols) - 1 - blackDepth);
	std::cout << "average updated: " << average << " mm" << std::endl;
	
	//if depth average is inside the threshold return true
	if(average < (depthZ.second + threshold_mm) && average > (depthZ.second - threshold_mm) ){
	    goodDepth = true;
	    std::cout << "safe depth: " << depthZ.first << " with distance: " << depthZ.second << " mm" << std::endl; 
	}
    }
    else{
         std::cout << "\033[1;33m[WARNING!]---NO SAFE DEPTH FOUNDED. RESTART STEP. \033[0m" << std::endl; 
    }
    return goodDepth;
}

cv::Mat ImageProcessing::getDebugRgbImage(cv::Point pos_xy){
  cv::Mat debug = cv::imread("/home/luca/Scrivania/Grasping_software/config/temp/debug_temp.jpg", CV_LOAD_IMAGE_COLOR);
  //draw point on image as a 5x5 white grid
  int gridRows = 5;
  int gridCols = 5;
  int offsetGrid = 2;
  for(int rows=0; rows < gridRows; ++rows)
  {
      for(int cols=0; cols < gridCols; ++cols)
      {
	  debug.at<cv::Vec3b>(cv::Point(pos_xy.x - offsetGrid + cols,  pos_xy.y - offsetGrid + rows))[0] = 0;
	  debug.at<cv::Vec3b>(cv::Point(pos_xy.x - offsetGrid + cols,  pos_xy.y - offsetGrid + rows))[1] = 0;
	  debug.at<cv::Vec3b>(cv::Point(pos_xy.x - offsetGrid + cols,  pos_xy.y - offsetGrid + rows))[2] = 255;
      }
  }
  debug.at<cv::Vec3b>(pos_xy)[0] = 0;
  debug.at<cv::Vec3b>(pos_xy)[1] = 0;
  debug.at<cv::Vec3b>(pos_xy)[2] = 255;
  return debug;
}