import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from keras.utils import *
from keras import optimizers
import tensorflow as tf

from keras import backend as K
from keras.models import model_from_json
import numpy as np
import cv2 
import XVAC2Numpy

BATCH_SIZE = 4000
IMAGE_H, IMAGE_W = 424, 424
GRID_H,  GRID_W  = 8 , 8
GRID_W_px = (IMAGE_W / GRID_W)
GRID_H_px = (IMAGE_H / GRID_H)

print("Welcome to the CNN output tester!")
print("Digit: 0 ---> Test with the ground truth dataset")
print("Digit: 1 ---> Test with not agnostic images not trained")
print("Digit: 2 ---> Test with agnostic images not trained")
print("Digit: 3 ---> Test with failed grasping ground truth images")

key = int(sys.stdin.read(1))
if(key == 0):
    #(X_train, Y_train, negative_samples, positive_samples, list_index_positive) = XVAC2Numpy.train_Only_Positive_Grasp_2Numpy(BATCH_SIZE)
    #(X_train, Y_train, negative_samples, positive_samples, list_index_positive) = XVAC2Numpy.train_Bilanced_Positive_And_Negative_Grasp_2Numpy(BATCH_SIZE)
    (X_train, Y_train, negative_samples, positive_samples, list_index) = XVAC2Numpy.train2Numpy(BATCH_SIZE)
    overall_samples = positive_samples + negative_samples
if(key > 0):
    (X_train, overall_samples) = XVAC2Numpy.testImages2Numpy(key)

# load json and create model
json_file = open('/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/my_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/weights/w-improvement-42-1.48.hdf5")
print("Loaded model from disk")
for j in range(overall_samples):
    if(key == 0):
        temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
        temp_img += str(j+1) + '.png'
    if(key == 1):
        temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/test/rgb_images/'
        temp_img += str(j+4001) + '.png'
    if(key == 2):
        temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/test/rgb_images/'
        temp_img += str(j+4051) + '.png'
    if(key == 3):
        temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/test/rgb_images/'
        temp_img += str(j+4101) + '.png'
    image = cv2.imread (temp_img)
    output = []
    output = loaded_model.predict(np.expand_dims(X_train[j], axis=0))
    pair_xy = []
    points_on_image = []
    grid_dim_x = output.shape[1]
    grid_dim_y = output.shape[2]
    list_max_scores = []
    list_max_x = []
    list_max_y = []
    list_max_grid_x = []
    list_max_grid_y = []

    temp_max = 1.0
    min_score = 1.0
    max_grid_x = 0
    max_grid_y = 0
    min_grid_x = 0
    min_grid_y = 0
    max_x = 0.0
    max_y = 0.0
    min_x = 0.0
    min_y = 0.0
    print(j)
    for k in range(4):    
        max_score = 0.0
        for i in range(grid_dim_x):
            for j in range(grid_dim_y):
                pair_xy = [output[0,i,j,0], output[0,i,j,1]]
                cv2.rectangle(image, (j*int(GRID_H_px), i*int(GRID_H_px)), (j*int(GRID_H_px)+int(GRID_H_px), i*int(GRID_H_px)+int(GRID_H_px)), (0,0,0), 1)
                
                draw1_x = int(j*int(GRID_H_px) + int((pair_xy[0]-0.5)*(2.0*float(GRID_W_px))))
                draw1_y = int(i*int(GRID_H_px)+int((pair_xy[1]-0.5)*(2.0*float(GRID_W_px))))
            
                if(output[0,i,j,4]<0.9):
                    cv2.circle(image,(draw1_x, draw1_y), 5, (0,0,255), -1)
                #find max_score
                if((output[0,i,j,4] > max_score) and output[0,i,j,4] < temp_max):
                    max_score = output[0,i,j,4]
                    max_x = pair_xy[0]
                    max_y = pair_xy[1]
                    max_grid_x = j
                    max_grid_y = i
                #find min_score
                if(output[0,i,j,4] < min_score):
                    min_score = output[0,i,j,4]
                    min_x = pair_xy[0]
                    min_y = pair_xy[1]
                    min_grid_x = j
                    min_grid_y = i
        list_max_x.append(max_x)
        list_max_y.append(max_y)
        list_max_grid_x.append(max_grid_x)
        list_max_grid_y.append(max_grid_y)
        list_max_scores.append(max_score)
        temp_max = max_score

    #draw the max_scores
    for i in range(len(list_max_scores)):
        draw_max_x = int(list_max_grid_x[i]*int(GRID_H_px) + int((list_max_x[i] - 0.5)*(2.0*float(GRID_H_px))))
        draw_max_y = int(list_max_grid_y[i]*int(GRID_H_px) + int((list_max_y[i] - 0.5)*(2.0*float(GRID_H_px))))
        #the higher max score predicted
        if(i == 0):
            cv2.circle(image,(draw_max_x, draw_max_y), 5, (255,0,0), -1)
            print("blue point: ", draw_max_x, draw_max_y, "with score: ", list_max_scores[i])    
        #the others
        if(i == 1):
            cv2.circle(image,(draw_max_x, draw_max_y), 5, (0,255,0), -1)
            print("green point: ", draw_max_x, draw_max_y, "with score: ", list_max_scores[i])  
        if(i == 2):
            cv2.circle(image,(draw_max_x, draw_max_y), 5, (0,255,255), -1)
            print("yellow point: ", draw_max_x, draw_max_y, "with score: ", list_max_scores[i])
        if(i == 3):
            cv2.circle(image,(draw_max_x, draw_max_y), 5, (2,106,253), -1)
            print("orange point: ", draw_max_x, draw_max_y, "with score: ", list_max_scores[i])    
    #Displayed the image
    cv2.imshow("Grasped Points", image)
    #print(output)
    #print(list_index)
    #print(max_score)
    cv2.waitKey(0)
    #print(max_score)

