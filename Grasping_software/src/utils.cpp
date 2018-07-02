#include "utils.h"
  
std::vector<boundingBox> Utils::getYOLOBoundingBoxes(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{  
   //boundingBox is a pair of cv::Point (first->topLeft,second->bottomRight)
    boundingBox temp; 
    std::vector<boundingBox> bBoxes;
    cv::Point topLeft;
    cv::Point bottomRight;
    int i,j;
    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class__ = -1;
        for(j = 0; j < classes; ++j){
	    if (probs[i][j] > thresh){
                if (class__ < 0) {
                    strcat(labelstr, names[j]);
                    class__ = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
            }
        }
        if(class__ >= 0){
            int width = im.h * .006;
            box b = boxes[i];
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
	    topLeft.x=left;
	    topLeft.y=top;
	    bottomRight.x=right;
	    bottomRight.y=bot;
	    temp.first=topLeft;
	    temp.second=bottomRight;
	    bBoxes.push_back(temp);   
        }
    }
    return bBoxes;
}

std::vector<boundingBox> Utils::getNormalizedBoundingBoxes(std::vector<boundingBox> bboxes)
{  
    std::vector<boundingBox> norm_bboxes;
    boundingBox temp;
    cv::Point topLeft;
    cv::Point bottomRight;
    for(int i; i < bboxes.size(); ++i)
    {
	topLeft.x = 550 + bboxes.at(i).first.x;
	topLeft.y = 300 + bboxes.at(i).first.y;
        bottomRight.x = 550 + bboxes.at(i).second.x;
	bottomRight.y = 300 + bboxes.at(i).second.y;
	temp.first = topLeft;
	temp.second = bottomRight;
	norm_bboxes.push_back(temp);
    }
    return norm_bboxes;
}  

void Utils::printBoundingBoxes( std::vector<boundingBox> bbox)
{
    for(int i=0; i<bbox.size();++i) std::cout << "TL " << bbox.at(i).first << " -- BR " << bbox.at(i).second << std::endl ;
}

cv::Mat Utils::convert4ChannelsToFloatImage(cv::Mat& im)
{ 
  cv::Mat float_depth(im.rows, im.cols, CV_32FC1, (float*)im.data);
  return float_depth;
}

float Utils::gen_random_float(float a, float b)
{
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}
