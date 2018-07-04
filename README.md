# myMasterThesis
Learning to grasp unknown objects with a robotic arm from experience

This repository contains the video, the latex files of my Master Thesis on Artificial Intelligence and Robotics (the pdf version and the slides about my presentation are available here -> https://drive.google.com/open?id=1ZfNXf_2pATPenSNPa_NihXHRey_p9ptv) and the two folders related to my project: Grasping_software and my_grasping_CNN, the first is a C++ acquisition software of the dataset, the second is related to the CNN used, with inside the python files for Tensorflow framework and dataset conversion.

To compile the Grasping_software you need to install in your machine: opencv-3.3.1 with CUDA=ON, the contributes opencv_contrib-3.3.1 and libfreenect2. You also need to download YOLO darknet from https://pjreddie.com/darknet/yolo/ and from Makefile export YOLO as libdarknet.so\libdarknet.a

To use the my_grasping_CNN you need to install Tensorflow framework from https://www.tensorflow.org/install/install_linux and Keras environment (sudo pip3 install keras). Before to train or test the network, you need to convert our dataset by executing the python file XVAC2Network.py. To train the 
CNN you need to execute the XVAC_cnn_train.py file. To test the Output of the Network, you need to execute test_my_cnn.py
