import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
from keras.datasets import cifar10
from keras.utils import *
from keras.optimizers import SGD
from keras import optimizers
import tensorflow as tf
import XVAC2Numpy

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AvgPool2D, LeakyReLU

from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

from keras import applications
from keras.layers import Input

from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import grids_gt_generator

IMAGE_H, IMAGE_W = 424, 424
GRID_H,  GRID_W  = 8 , 8
BATCH_IMG_TRAIN_SIZE = 4000
BATCH_IMG_VAL_SIZE = 0
negative_samples = 0
positive_samples = 0

print("Welcome to the XVAC_CNN training software!")
print("Digit: 0 ---> Fully the entire training dataset")
print("Digit: 1 ---> For train only positive samples dataset")
print("Digit: 2 ---> For train half positive samples and half negative samples")
print("Digit: 3 ---> For train 1/3 positive samples and 2/3 negative samples")

key = int(sys.stdin.read(1))
if(key == 0):
	(X_train, Y_train, negative_samples, positive_samples, index_pos) = XVAC2Numpy.train2Numpy(BATCH_IMG_TRAIN_SIZE)
if(key == 1):
	(X_train, Y_train, negative_samples, positive_samples, index_pos) = XVAC2Numpy.train_Only_Positive_Grasp_2Numpy(BATCH_IMG_TRAIN_SIZE)
if(key == 2):
	(X_train, Y_train, negative_samples, positive_samples, index_pos) = XVAC2Numpy.train_Half_Positive_Half_Negative_Grasp_2Numpy(BATCH_IMG_TRAIN_SIZE)
if(key == 3):
	(X_train, Y_train, negative_samples, positive_samples, index_pos) = XVAC2Numpy.train_Bilanced_Positive_And_Negative_Grasp_2Numpy(BATCH_IMG_TRAIN_SIZE)

total_number_samples = negative_samples + positive_samples
BATCH_IMG_TRAIN_SIZE = total_number_samples

(X_test, Y_test) = XVAC2Numpy.val2Numpy(BATCH_IMG_VAL_SIZE)
#generate the y_batch
Y_train = grids_gt_generator.grids_batch_generator(Y_train, 0, IMAGE_H, IMAGE_W, GRID_H, GRID_W, BATCH_IMG_TRAIN_SIZE, BATCH_IMG_VAL_SIZE)

#Y_test = grids_gt_generator.grids_batch_generator(Y_test, 1, IMAGE_H, IMAGE_W, GRID_H, GRID_W, BATCH_IMG_TRAIN_SIZE, BATCH_IMG_VAL_SIZE)


input_shape = (424, 424, 4)

#modified CNN
base_model = Sequential([
    Conv2D(16, (7, 7), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(128, (5, 5), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(256, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    BatchNormalization(),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    Conv2D(5, (5, 5), activation='relu', padding='same',),
    AvgPool2D(pool_size=(4, 4), strides=(4, 4)),
    BatchNormalization(),
    Flatten(),
    Dense(8*8*5, activation='relu'),
    Dense(8*8*5, activation='sigmoid'),
    Reshape((8,8,5))
])

def custom_loss(y_true, y_pred):
	#some filters on the GT and prediction of the CNN
    true_x = y_true[:,:,:,0]
    true_y = y_true[:,:,:,1]
    true_roll = y_true[:,:,:,2]
    true_pitch = y_true[:,:,:,3]
    true_confidence = y_true[:,:,:,4] 
    
    true_mask=true_x
    true_mask = tf.add(true_x,true_y)
    true_mask = tf.add(true_mask,true_roll)
    true_mask = tf.add(true_mask,true_pitch)
    true_mask = tf.add(true_mask,true_confidence)

    pred_x = y_pred[:,:,:,0]
    pred_y = y_pred[:,:,:,1]
    pred_roll = y_pred[:,:,:,2]
    pred_pitch = y_pred[:,:,:,3]
    pred_confidence = y_pred[:,:,:,4]

    
    #################THE LOSS CONFIDENCE#################
    diff_confidence = tf.square(pred_confidence - true_confidence)
    #filter the groundtruth_cell for the value of the confidence
    confidence_mask = tf.greater(true_mask, 0.0)
    #take the confidence for that cell
    confidence = tf.boolean_mask(diff_confidence, confidence_mask)
    
    #take the confidence groundtruth
    confidence_gt = tf.boolean_mask(true_confidence, confidence_mask) 
   
   	#check the GT confidence for setting lambda
    lambda_neg_confidence = 0.0
    if(negative_samples > 0):
    	#lambda_neg_confidence = 1.0 - ((float(negative_samples)/float(total_number_samples) + 0.1)) 
    	lambda_neg_confidence = 1.0 - (float(negative_samples)/float(total_number_samples))
    lambda_pos_samples = (1.0 - lambda_neg_confidence)
    condition = tf.greater(confidence_gt, 0.0)
    lambda_grasp_nograsp = tf.where(condition, (tf.multiply(tf.divide(confidence,confidence),lambda_pos_samples)),tf.multiply(tf.divide(confidence,confidence),lambda_neg_confidence))
    lambda_grasp_nograsp = lambda_grasp_nograsp[0]
    #reduce sum for that loss
    loss_confidence = lambda_grasp_nograsp * tf.reduce_sum(confidence)
    #loss_confidence = tf.reduce_sum(confidence)
    

    #################THE LOSS ROLL-PITCH#################
    diff_angles = tf.square(pred_roll - true_roll) + tf.square(pred_pitch - true_pitch)
    angles_mask = tf.greater(true_mask, 0.0)
    angles_maskered = tf.boolean_mask(diff_angles, angles_mask)
    lambda_orientation = 1.0
    #reduce sum for that loss
    loss_orient = lambda_orientation * tf.reduce_sum(angles_maskered)
	

    #################THE LOSS XY#################
    #filter the groundtruth_cell for the xy position
    diff = tf.square(pred_x - true_x) + tf.square(pred_y - true_y)
    mask = tf.greater(true_mask, 0.0)
    #take the shift position
    maskered = tf.boolean_mask(diff, mask)    
    #the xy regularization factor 
    lambda_xy = 1.0
    loss_xy = lambda_xy * tf.reduce_sum(maskered)
    




    #the overall loss
    loss = loss_xy + loss_orient + loss_confidence
  
    #loss = tf.Print(loss, [lambda_neg_confidence], "lambda_neg_confidence", summarize=100000)
    #loss = tf.Print(loss, [lambda_pos_samples], "lambda_pos_samples", summarize=100000)
    
    #loss = tf.Print(loss, [pred_confidence], " pred_confidence", summarize=100000)
    #loss = tf.Print(loss, [pred_x], " pred_x", summarize=100000)
    #loss = tf.Print(loss, [tf.shape(confidence)], " confidence shape", summarize=100000)
    #loss = tf.Print(loss, [confidence], " confidence", summarize=100000)
    #loss = tf.Print(loss, [confidence_gt], " confidence_gt", summarize=100000)
    #loss = tf.Print(loss, [lambda_grasp_nograsp], " lambda_grasp_nograsp", summarize=100000)
    #loss = tf.Print(loss, [loss_xy], "loss_xy", summarize=100000)
    #loss = tf.Print(loss, [pred_confidence], "pred_confidence", summarize=100000)

    return loss

#load pretrained weights
#base_model.load_weights("/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/weights/w-improvement-40-1.94.hdf5")

optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
base_model.compile(loss=custom_loss, optimizer=optimizer)

# checkpoint
filepath='/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/weights/w-improvement-{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
# Fit the model

base_model.fit(X_train, Y_train, nb_epoch=100, batch_size=16, callbacks=callbacks_list, verbose=1)

# serialize model to JSON
model_json = base_model.to_json()
with open("/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/my_model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
base_model.save_weights("/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/weights/model.h5")
print("Saved model to disk")

#base_model.evaluate(X_test, Y_test, batch_size=BATCH_IMG_VAL_SIZE, verbose=1)

#a = base_model.get_weights()
#print(a)

def get_weights():
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]



