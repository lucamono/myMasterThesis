#include "datasetInterface.h"

std::string::size_type sz;   // alias of size_t

int DatasetInterface::getCurrentScenario(){
    int scenario=-1;
    std::ifstream loadDataset_file("/home/luca/Scrivania/XVAC_Dataset/resumeGrasping.txt");
    std::string lastScenario = "No resume scenario available";
    if (loadDataset_file){
	    std::string line;
	    while (getline (loadDataset_file,line)){
		    loadDataset_file.ignore(0);
		    std::string delimiter = "_";
		    int pos = line.find(delimiter);
		    std::string token;
		    while (pos !=-1) {
			    token = line.substr(0, pos);
			    lastScenario = token;
			    line.erase(0, pos + delimiter.length());
			    pos = line.find(delimiter);
		    }
	    }
	}
    scenario = std::stoi (lastScenario,&sz);
    std::cout << "resume at: " << lastScenario << std::endl; 
    return scenario;
}

void DatasetInterface::updateDataset(int scenario, cv::Mat image_Reg_rgb, cv::Mat image_Reg_depth, cv::Mat image_NoReg_rgb, cv::Mat image_NoReg_depth,  cv::Mat debug_image, Eigen::Vector3f eigenPoint3D, cv::Point posReg_xy, cv::Point posNoReg_xy, float depth, float pitch, float yaw, bool success){
    std::string str_path = "mkdir -p /home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp";
    const char* command_path = str_path.c_str();
    std::string rgb_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_rgb_image.png";
    std::string depth_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_depth_image.png";
    std::string debug_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_debug.png";
    std::string rgb_NoReg_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_rgb_NoReg_image.png";
    std::string depth_NoReg_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_depth_NoReg_image.png";
    FILE *fd;
    std::string file_path = "/home/luca/Scrivania/XVAC_Dataset/"+std::to_string(scenario)+"_grasp/"+std::to_string(scenario)+"_data.txt";
    //create the rootFolder of the dataset
    if(scenario ==1){
	const int dir_err = system("mkdir -p /home/luca/Scrivania/XVAC_Dataset");
	if (-1 == dir_err)
	{
	    printf("Error creating directory!n");
	exit(1);
	}
	//create the first subfolder
	const int dir_err2 = system(command_path);
	if (-1 == dir_err2)
	{
	    printf("Error creating directory!n");
	exit(1);
	}
	//save the rgb image acquired
	cv::imwrite(rgb_path,image_Reg_rgb);
	//save the depth image acquired
	cv::imwrite(depth_path,image_Reg_depth);
	//save the debug image 
	cv::imwrite(debug_path,debug_image);
	//save the NoReg rgb image 
	cv::imwrite(rgb_NoReg_path,image_NoReg_rgb);
	//save the NoReg depth image 
	cv::imwrite(depth_NoReg_path,image_NoReg_depth);
    }
    else{
	//create the other subfolders
	const int dir_err3 = system(command_path);
	if (-1 == dir_err3)
	{
	    printf("Error creating directory!n");
	exit(1);
	}
	//save the rgb image acquired
	cv::imwrite(rgb_path,image_Reg_rgb);
	//save the depth image acquired
	cv::imwrite(depth_path,image_Reg_depth);
	//save the debug image 
	cv::imwrite(debug_path,debug_image);
	//save the NoReg rgb image 
	cv::imwrite(rgb_NoReg_path,image_NoReg_rgb);
	//save the NoReg depth image 
	cv::imwrite(depth_NoReg_path,image_NoReg_depth);
    }
    //write mode file
    fd=fopen(file_path.c_str(), "w");
    if( fd==NULL )
    {
	perror("Error I/O");
	exit(1);
    }
    //queue: world3Dpos_x world3Dpos_y world3Dpos_z posReg_x posReg_y posNoReg_x posNoReg_y depth pitch yaw graspingOnVacuumBoolean
    fprintf(fd, "%f %f %f %d %d %d %d %f %f %f %s", eigenPoint3D(0), eigenPoint3D(1), eigenPoint3D(2), posReg_xy.x, posReg_xy.y, posNoReg_xy.x, posNoReg_xy.y, depth, pitch, yaw, success?"true":"false");
    // close the file
    fclose(fd);
}

void DatasetInterface::updateResumeState(int scenario){
    FILE *fd;
    std::string path_dataset = "/home/luca/Scrivania/XVAC_Dataset/resumeGrasping.txt";
    std::string resumeLabel = std::to_string(scenario) + "_grasp";
    //write mode file
    fd=fopen(path_dataset.c_str(), "w");
    if( fd==NULL )
    {
	perror("Error I/O");
	exit(1);
    }
    fprintf(fd, "%s", resumeLabel.c_str());
    // close the file
    fclose(fd);
}

int DatasetInterface::checkInputKey(char inp){
    int countScenario;
    if(inp == 'l'){
	countScenario = getCurrentScenario(); 
    }
    else{
	if(inp == 'n'){
	    std::cout << "creation of new dataset available" << std::endl;
	    countScenario = 1;
	}
	else{
	    std::cout << "wrong input inserted, exit program" << std::endl;
	    countScenario = -1;
	}
    }
    return countScenario;
}

int DatasetInterface::initDataset(){
    char inp;
    int countScenario ;
    std::cout << "Press l to load dataset or n for create new dataset:" << std::endl;
    std::cin >> inp;
    countScenario = checkInputKey(inp);
    return countScenario;
}