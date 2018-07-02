#include "imageProcessing.h"

extern "C"
{
#include "darknet.h"
}


class yoloDetector {
    
    private:
        Utils u;
	ImageProcessing im;
	float thresh;
	float hier_thresh; 
	char **names;
	int fullscreen;
	char *outfile;
	image **alphabet;
	network *net;
	
    public:
	yoloDetector(){};
	void initDarknetParameters();
	std::vector<boundingBox> run_yolo_detector(cv::Mat loaded_image);
};
