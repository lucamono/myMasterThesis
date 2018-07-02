import numpy as np
import os
import cv2 
import PIL
from PIL import Image

#create numpy array from training data
def train2Numpy(number_train):
	print("loading XVAC training dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/train/rgb_images'
	X_data = []
	Y_data = []
	folder_size = number_train
	list_index = []
	k = 1
	positive_samples = 0
	negative_samples = 0
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/depth_images/'
	temp_annotation = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
	for j in range(folder_size):
		
		#the rgb-image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		
		#the depth-image part
		temp_depth = temp_depth_img
		temp_depth += str(k) + '.png'
		image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
		depth_image = np.frombuffer(image_depth, np.float32)
		depth_image.shape = (424, 424, 1)
		#depth_image = np.flipud(depth_image)
		depth_image = depth_image/1200.00
		idx = depth_image[:,:,0] > 1.0
		depth_image[idx,0] = 1.0
		image = np.concatenate([image, depth_image], axis=2)
		
		X_data.append(image)
		
		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
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
			negative_samples = negative_samples + 1
		else:
			tmp_arr.append(1.0)
			positive_samples = positive_samples + 1
		list_index.append(k)
		k = k+1	
		Y_data.append(tmp_arr)
	print("Numpy array created succesfully", positive_samples, negative_samples)
	return (np.array(X_data) , np.array(Y_data), negative_samples, positive_samples, list_index)

def testImages2Numpy(value):
	print("loading XVAC testing dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/test/rgb_images'
	X_data = []
	Y_data = []

	#200 it's a good choice
	folder_size = 50
	if(value == 1):
		k = 4001
	if(value == 2):
		k = 4051
	if(value == 3):
		k = 4101
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/test/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/test/depth_images/'

	for j in range(folder_size):
		
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		#the depth-image part
		temp_depth = temp_depth_img
		temp_depth += str(k) + '.png'
		image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
		depth_image = np.frombuffer(image_depth, np.float32)
		depth_image.shape = (424, 424, 1)
		#depth_image = np.flipud(depth_image)
		depth_image = depth_image/1200.00
		idx = depth_image[:,:,0] > 1.0
		depth_image[idx,0] = 1.0
		image = np.concatenate([image, depth_image], axis=2)

		X_data.append(image)
		k = k+1	
	print("Numpy array created succesfully")
	return (np.array(X_data), folder_size)


def val2Numpy(number_val):
	print("loading XVAC testing dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/test/rgb_images'
	X_data = []
	Y_data = []

	#200 it's a good choice
	folder_size = number_val
	k = 1
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/val/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/val/depth_images/'
	temp_annotation = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/val/annotations/'
	for j in range(folder_size):
		
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		#the depth-image part
		temp_depth = temp_depth_img
		temp_depth += str(k) + '.png'
		image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
		depth_image = np.frombuffer(image_depth, np.float32)
		depth_image.shape = (424, 424, 1)
		#depth_image = np.flipud(depth_image)
		depth_image = depth_image/1200.00
		idx = depth_image[:,:,0] > 1.0
		depth_image[idx,0] = 1.0
		image = np.concatenate([image, depth_image], axis=2)

		X_data.append(image)
		
		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
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
		k = k+1	
		Y_data.append(tmp_arr)
	print("Numpy array created succesfully")
	return (np.array(X_data) , np.array(Y_data))



def train_Only_Positive_Grasp_2Numpy(number_train):
	print("loading XVAC training dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/train/rgb_images'
	X_data = []
	Y_data = []
	folder_size = number_train
	list_index_pos = []
	k = 1
	positive_samples = 0
	negative_samples = 0
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/depth_images/'
	temp_annotation = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
	for j in range(folder_size):
		
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
			if(cont==3 or cont ==4):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			if(cont == 8 or cont == 9):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			array = array[array.find(' ')+1:len(array)]
			cont = cont + 1
			
		if (array[0] == 't'):
			tmp_arr.append(1.0)

			#the depth-image part
			temp_depth = temp_depth_img
			temp_depth += str(k) + '.png'
			image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
			depth_image = np.frombuffer(image_depth, np.float32)
			depth_image.shape = (424, 424, 1)
			#depth_image = np.flipud(depth_image)
			depth_image = depth_image/1200.00
			idx = depth_image[:,:,0] > 1.0
			depth_image[idx,0] = 1.0
			image = np.concatenate([image, depth_image], axis=2)

			X_data.append(image)
			Y_data.append(tmp_arr)
			positive_samples = positive_samples + 1
			list_index_pos.append(k)
		k = k+1	
		
	print("Numpy array created succesfully",positive_samples,negative_samples)
	return (np.array(X_data) , np.array(Y_data), negative_samples, positive_samples, list_index_pos)


def train_Half_Positive_Half_Negative_Grasp_2Numpy(number_train):
	print("loading XVAC training dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/train/rgb_images'
	X_data = []
	Y_data = []
	folder_size = number_train
	list_index = []
	k = 1
	positive_samples = 0
	negative_samples = 0
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/depth_images/'
	temp_annotation = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
	#take the positive grasping samples
	for j in range(folder_size):
		
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		
		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
			if(cont==3 or cont ==4):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			if(cont == 8 or cont == 9):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			array = array[array.find(' ')+1:len(array)]
			cont = cont + 1
			
		if (array[0] == 't'):
			tmp_arr.append(1.0)

			#the depth-image part
			temp_depth = temp_depth_img
			temp_depth += str(k) + '.png'
			image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
			depth_image = np.frombuffer(image_depth, np.float32)
			depth_image.shape = (424, 424, 1)
			#depth_image = np.flipud(depth_image)
			depth_image = depth_image/1200.00
			idx = depth_image[:,:,0] > 1.0
			depth_image[idx,0] = 1.0
			image = np.concatenate([image, depth_image], axis=2)

			X_data.append(image)
			Y_data.append(tmp_arr)
			positive_samples = positive_samples + 1
			list_index.append(k)
		k = k+1	

	#take the same number of false grasping samples
	k = 1
	for j in range(folder_size):
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
			if(cont==3 or cont ==4):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			if(cont == 8 or cont == 9):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			array = array[array.find(' ')+1:len(array)]
			cont = cont + 1
			
		if (array[0] == 'f' and (negative_samples < positive_samples)):
			tmp_arr.append(0.0)

			#the depth-image part
			temp_depth = temp_depth_img
			temp_depth += str(k) + '.png'
			image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
			depth_image = np.frombuffer(image_depth, np.float32)
			depth_image.shape = (424, 424, 1)
			#depth_image = np.flipud(depth_image)
			depth_image = depth_image/1200.00
			idx = depth_image[:,:,0] > 1.0
			depth_image[idx,0] = 1.0
			image = np.concatenate([image, depth_image], axis=2)

			X_data.append(image)
			Y_data.append(tmp_arr)
			negative_samples = negative_samples + 1
			list_index.append(k)
		k = k+1	
	print("Numpy array created succesfully",positive_samples,negative_samples)
	return (np.array(X_data) , np.array(Y_data), negative_samples, positive_samples, list_index)

def train_Bilanced_Positive_And_Negative_Grasp_2Numpy(number_train):
	print("loading XVAC training dataset...")
	#root_directory = 'XVAC2NETWORK'
	#train_rgb_directory = root_directory + '/train/rgb_images'
	X_data = []
	Y_data = []
	folder_size = number_train
	list_index = []
	k = 1
	positive_samples = 0
	negative_samples = 0
	temp_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
	temp_depth_img = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/depth_images/'
	temp_annotation = '/media/luca/SSD500GB/Ubuntu/TESI_DEFINITIVO/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
	#take the positive grasping samples
	for j in range(folder_size):
		
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		
		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
			if(cont==3 or cont ==4):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			if(cont == 8 or cont == 9):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			array = array[array.find(' ')+1:len(array)]
			cont = cont + 1
			
		if (array[0] == 't'):
			tmp_arr.append(1.0)

			#the depth-image part
			temp_depth = temp_depth_img
			temp_depth += str(k) + '.png'
			image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
			depth_image = np.frombuffer(image_depth, np.float32)
			depth_image.shape = (424, 424, 1)
			#depth_image = np.flipud(depth_image)
			depth_image = depth_image/1200.00
			idx = depth_image[:,:,0] > 1.0
			depth_image[idx,0] = 1.0
			image = np.concatenate([image, depth_image], axis=2)

			X_data.append(image)
			Y_data.append(tmp_arr)
			positive_samples = positive_samples + 1
			list_index.append(k)
		k = k+1	

	#take the 2xnumbers of false grasping samples
	k = 1
	for j in range(folder_size):
		#the image part
		temp_rgb = temp_img
		temp_rgb += str(k) + '.png'
		image = cv2.imread (temp_rgb)
		cv2.normalize(image,image, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		#the annotation part
		temp_ann = temp_annotation
		temp_ann += str(k) + '.txt'
		array = []
		with open(temp_ann, "r") as ins:
			for line in ins:
				array.append(line)
		
		index = 0
		#start parse
		array = str(array)
		array = array[array.find('[')+2:len(array)-2]
		value=float(array[0:array.find(' ')])
		cont = 0
		tmp_arr = []
		while(cont < 10):
			#index = value
			if(cont==3 or cont ==4):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			if(cont == 8 or cont == 9):
				value=float(array[0:array.find(' ')])
				tmp_arr.append(value)
			array = array[array.find(' ')+1:len(array)]
			cont = cont + 1
			
		if (array[0] == 'f' and (negative_samples < 2*positive_samples)):
			tmp_arr.append(0.0)

			#the depth-image part
			temp_depth = temp_depth_img
			temp_depth += str(k) + '.png'
			image_depth = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED)
			depth_image = np.frombuffer(image_depth, np.float32)
			depth_image.shape = (424, 424, 1)
			#depth_image = np.flipud(depth_image)
			depth_image = depth_image/1200.00
			idx = depth_image[:,:,0] > 1.0
			depth_image[idx,0] = 1.0
			image = np.concatenate([image, depth_image], axis=2)

			X_data.append(image)
			Y_data.append(tmp_arr)
			negative_samples = negative_samples + 1
			list_index.append(k)
		k = k+1	
	print("Numpy array created succesfully",positive_samples,negative_samples)
	return (np.array(X_data) , np.array(Y_data), negative_samples, positive_samples, list_index)		

def normalizePixel(x):
	x_max = 424
	x_min = 1
	return (x-x_min)/(x_max-x_min)

def normalizeAngleRoll(x):
	x_max = 17
	x_min = -17
	return ((x-x_min)/(x_max-x_min))

def normalizeAnglePitch(x):
	if(x > 0):
		x_max = 180
		x_min = 163	
		res = 0.5*(x-x_min)/(x_max-x_min)
	else:
		x_max = 163
		x_min = 180	
		res = 0.5 + 0.5*(abs(x)-x_min)/(x_max-x_min)
	return res
