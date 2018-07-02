
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from keras.utils import *
from keras import optimizers
import tensorflow as tf

from keras import backend as K
from keras.models import model_from_json
import numpy as np
import XVAC2Numpy
import cv2 

IMAGE_H, IMAGE_W = 424, 424
GRID_H,  GRID_W  = 8 , 8
BATCH_SIZE = 2484

GRID_W_px = (IMAGE_W / GRID_W)
GRID_H_px = (IMAGE_H / GRID_H)

def batch_generator(YVAC_data):
    #case of training data
    y_batch = np.zeros((1, GRID_H, GRID_W, 5)) 
    groun_truth_x = YVAC_data[0][0]
    groun_truth_y = YVAC_data[0][1]
    roll = YVAC_data[0][2]
    pitch = YVAC_data[0][3]
    grasp = YVAC_data[0][4]
    pos_x_tmp = groun_truth_x / (float(IMAGE_W) / GRID_W)
    pos_y_tmp = groun_truth_y / (float(IMAGE_H) / GRID_H)
    grid_x = int(np.floor(pos_x_tmp))
    grid_y = int(np.floor(pos_y_tmp))

    if(grid_x*GRID_W_px < groun_truth_x):
        shift_x = (groun_truth_x - grid_x*GRID_W_px) / float(IMAGE_W)
    else:
        shift_x = (grid_x*GRID_W_px - groun_truth_x) / float(IMAGE_W)

    if(grid_y*GRID_H_px < groun_truth_y):
        shift_y = (groun_truth_y - grid_y*GRID_H_px) / float(IMAGE_W)
    else:
        shift_y = (grid_y*GRID_H_px - groun_truth_y) / float(IMAGE_W)

    y_batch[0, grid_y, grid_x, 0] = shift_x
    y_batch[0, grid_y, grid_x, 1] = shift_y
    y_batch[0, grid_y, grid_x, 2] = XVAC2Numpy.normalizeAngleRoll(roll)
    y_batch[0, grid_y, grid_x, 3] = XVAC2Numpy.normalizeAnglePitch(pitch)
    y_batch[0, grid_y, grid_x, 4] = grasp

    return (y_batch, grid_x, grid_y, groun_truth_x, groun_truth_y)


temp_img = '/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
temp_annotation = '/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
k = 1
for i in range(BATCH_SIZE):
	#load rgb image
	temp_rgb = temp_img
	temp_rgb += str(k) + '.png'
	image = cv2.imread (temp_rgb)
	X_data = []
	X_data.append(image)
	X_test = np.array(X_data)

	#load annotation
	temp_ann = temp_annotation
	temp_ann += str(k) + '.txt'
	array = []
	Y_data = []
	with open(temp_ann, "r") as ins:
		for line in ins:
			array.append(line)
	#start parse
	array = str(array)
	array = array[array.find('[')+2:len(array)-2]
	value=float(array[0:array.find(' ')])
	cont = 0
	tmp_arr = []
	while(cont < 10):
		if(cont==3 or cont ==4):
			value=float(array[0:array.find(' ')])
			tmp_arr.append(value)
		if(cont == 8 or cont == 9):
			value=float(array[0:array.find(' ')])
			tmp_arr.append(value)
		array = array[array.find(' ')+1:len(array)]
		cont = cont + 1
		
	if (array[0] == 'f'):
		tmp_arr.append(0.0)
	else:
		tmp_arr.append(1.0)
	
	Y_data.append(tmp_arr)
	Y_test = np.array(Y_data)

	ground_truth, grid_x, grid_y, groun_truth_x, groun_truth_y = batch_generator(Y_test)

	shift_x = ground_truth[0, grid_y, grid_x, 0]
	shift_y = ground_truth[0, grid_y, grid_x, 1]
	graspOrNot = ground_truth[0, grid_y, grid_x, 4]
	cv2.rectangle(image, (grid_x*int(GRID_H_px), grid_y*int(GRID_H_px)), (grid_x*int(GRID_H_px)+int(GRID_H_px), grid_y*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 1)
	cv2.circle(image,(grid_x*int(GRID_H_px) + int(shift_x*IMAGE_W), grid_y*int(GRID_H_px)+int(shift_y*IMAGE_W)), 1, (0,255,0), -1)
	#Displayed the image
	cv2.imshow("Grasped Points", image)
	print("batch_size: ", k)
	print("Cell grid: ", grid_x, grid_y, "index x: ", grid_x*int(GRID_H_px), "index_y: ", grid_y*int(GRID_H_px))
	print("Shift: ", shift_x*IMAGE_W, shift_y*IMAGE_W)
	print("Point of Grasping: ", groun_truth_x, groun_truth_y)
	print("Grasped: ", graspOrNot)
	cv2.waitKey(0)
	k = k+1	
