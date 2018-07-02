#include "yoloDetector.h"

void yoloDetector::initDarknetParameters()
{
    char *datacfg = "/home/luca/Scrivania/Grasping_software/darknet/cfg/flexsight-dataset.data";
    lista *options = read_data(datacfg);
    char *name_list = option_find_str_lista(options, "names", "data/names.list");
    names = get_labels(name_list);
    char *cfgfile = "/home/luca/Scrivania/Grasping_software/darknet/cfg/yolo-voc.2.0_flexsight.cfg";
    char *weightfile = "/home/luca/Scrivania/Grasping_software/darknet/yolo-voc_1000.weights";
    thresh = 0.005;
    hier_thresh = 0.5;
    fullscreen = 0;
    alphabet = (image**)load_alphabet();
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    outfile;
}

std::vector<boundingBox> yoloDetector::run_yolo_detector(cv::Mat loaded_image)
{
    //prepare image for yolo (crop the full-hd image)
    
    //optimize the image for yolo detection 
    cv::Mat masked_image = im.applyMask(loaded_image);
    //crop the image
    cv::Mat cropped_image = im.cropImage(masked_image);
    //save image on file
    cv::imwrite("/home/luca/Scrivania/Grasping_software/config/temp/crop_image.jpg",cropped_image);
    
    //load the cropped image
    char *filename = "/home/luca/Scrivania/Grasping_software/config/temp/crop_image.jpg";
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.3;
    std::vector<boundingBox> bBoxes;
    while(1){
	  strncpy(input, filename, 256);
	  image im = load_image_color(input,0,0);
	  image sized = letterbox_image(im, net->w, net->h);
	  layer l = net->layers[net->n-1];
	  box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
	  float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
	  for(j = 0; j < l.w*l.h*l.n; ++j) 
	      probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
	  float **masks = 0;
	  if (l.coords > 4){
	      masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
	      for(j = 0; j < l.w*l.h*l.n; ++j) 
		  masks[j] = (float*)calloc(l.coords-4, sizeof(float *));
	  }
	  float *X = sized.data;
	  time=what_time_is_it_now();
	  network_predict(net, X);
	  printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
	  get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
	  //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	  if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	  draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
	  //get the bounding boxes from yolo
	  bBoxes=u.getYOLOBoundingBoxes(im, l.w*l.h*l.n, thresh, boxes, probs, names, l.classes);
	  free_image(im);
	  free_image(sized);
	  free(boxes);
	  free_ptrs((void **)probs, l.w*l.h*l.n);
	  if (filename) break;
    }
    //translate the bounding boxes in the original image 
    std::vector<boundingBox> bBoxes_norm;
    bBoxes_norm=u.getNormalizedBoundingBoxes(bBoxes);
    u.printBoundingBoxes(bBoxes_norm);
    cv::Mat final_output = im.drawBbox(loaded_image, bBoxes_norm);
    cv::imshow( "ACQUIRED IMAGE_RGB", final_output);
    return bBoxes_norm;
}