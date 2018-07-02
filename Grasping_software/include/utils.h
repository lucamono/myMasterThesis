#include <pcl/io/pcd_io.h>
#include <datasetInterface.h>

extern "C"
{
#include "darknet.h"  
}

static int num_s, num_f;

typedef std::pair<cv::Point, cv::Point> boundingBox;

class Utils {
  
    public:
	Utils(){};
	std::vector<boundingBox> getYOLOBoundingBoxes(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes);
	void printBoundingBoxes( std::vector<boundingBox> bbox);
	std::vector<boundingBox> getNormalizedBoundingBoxes(std::vector<boundingBox> bboxes);
	cv::Mat convert4ChannelsToFloatImage(cv::Mat& im);
	float gen_random_float(float min, float max);
};
