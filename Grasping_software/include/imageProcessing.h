#include "utils.h"

typedef std::pair<cv::Point, cv::Point> boundingBox;

class ImageProcessing {
 
    public:
	ImageProcessing(){};
	cv::Mat applyMask(cv::Mat image);
	cv::Mat cropImage(cv::Mat image);
	cv::Mat drawBbox(cv::Mat image, std::vector<boundingBox> bBoxes);
	cv::Mat filterBbox(cv::Mat image, std::vector<boundingBox> bBoxes);
	std::pair<cv::Mat,cv::Point2f> selectRandomPointGrasp(cv::Mat image,  cv::Mat kinect_image, std::vector<boundingBox> bboxes);
	std::pair <cv::Point2f,float> getDepthPoint(cv::Mat rgb_registered, cv::Mat depth_registered);
	bool checkDepthIsSafe(cv::Mat depth_registered, std::pair <cv::Point2f,float> depthZ);
	cv::Mat getDebugRgbImage(cv::Point pos_xy);
};
