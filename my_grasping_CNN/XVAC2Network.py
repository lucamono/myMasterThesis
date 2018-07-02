import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import shutil
import os
import cv2

#create directories
root_directory = 'XVAC2NETWORK'
val_rgb_directory = root_directory + '/val/rgb_images'
val_depth_directory = root_directory + '/val/depth_images'
val_annotation_directory = root_directory + '/val/annotations'
train_rgb_directory = root_directory + '/train/rgb_images'
train_depth_directory = root_directory + '/train/depth_images'
train_annotation_directory = root_directory + '/train/annotations'
test_rgb_directory = root_directory + '/test/rgb_images'
test_depth_directory = root_directory + '/test/depth_images'

if not os.path.exists(root_directory):
    os.makedirs(root_directory)
if not os.path.exists(train_rgb_directory):
    os.makedirs(train_rgb_directory)
if not os.path.exists(train_depth_directory):
    os.makedirs(train_depth_directory)
if not os.path.exists(train_annotation_directory):
    os.makedirs(train_annotation_directory)

print("Welcome to the XVAC2Network converter!")
print("Digit: 0 ---> Fully training dataset without validation")
print("Digit: 1 ---> For split into training and validation dataset")
print("Digit: 2 ---> For create image test dataset")

#read file config dataset
acquisition_size_path = '/media/luca/TERZIARIO/ubuntu_data/dataset_luca/XVAC_Dataset/resumeGrasping.txt'
fd = os.open(acquisition_size_path,os.O_RDWR)
myString = os.read(fd,100)
#the number of the dataset's scenarious
acquisition_size = int(myString[0:myString.find('_')]) - 1

key = int(sys.stdin.read(1))
if(key == 0):
	print("Creation of the training set...\n")
	k=1
	printed = True
	temp = '/media/luca/TERZIARIO/ubuntu_data/dataset_luca/XVAC_Dataset/'
	for j in range(acquisition_size):
		temp_rgb = temp
		temp_depth = temp
		temp_annot = temp
		
		temp_rgb+=str(k) + '_grasp/' + str(k) + '_rgb_image.png'
		temp_depth+=str(k) + '_grasp/' + str(k) + '_depth_image.png'
		temp_annot+=str(k) + '_grasp/' + str(k) + '_data.txt'
		
		shutil.copyfile(temp_rgb, os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png')	
		shutil.copyfile(temp_depth, os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png')
		shutil.copyfile(temp_annot, os.getcwd() + '/' + train_annotation_directory + '/' + str(k) + '.txt')
		#rgb crop process
		img = cv2.imread(os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png')
		crop_img = img[0:424, 0:424]
		cv2.imwrite(os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png', crop_img)
		#depth crop process
		img = cv2.imread(os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png',cv2.IMREAD_UNCHANGED)
		crop_img = img[0:424, 0:424]
		cv2.imwrite(os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png', crop_img)
		k=k+1
		
		if(float(acquisition_size/4) > (j) and printed):
			print("progress [=>    ]20%")
			printed = False
		if(float(acquisition_size/3) > (j) and float(acquisition_size/(4)) < (j) and (not(printed))):
			print("progress [==>   ]40%")
			printed = True
		if(float(acquisition_size/2) > (j) and float(acquisition_size/3) < (j) and printed):
			print("progress [===>  ]60%")
			printed = False
		if(float(acquisition_size/1.5) > (j) and float(acquisition_size/2) < (j) and (not(printed))):
			print("progress [====> ]80%")
			printed = True
	print("progress [=====>]100%\n")	
	print("[OK] Succesfully created")

if(key == 1):

	print("Insert the size of the validation set (max 999) and press enter: (Overall DatasetSize = " + str(acquisition_size) + ")")
	key2 = int(sys.stdin.read(4))
	print("Creation of the validation set...\n")

	if not os.path.exists(val_rgb_directory):
	    os.makedirs(val_rgb_directory)
	if not os.path.exists(val_depth_directory):
	    os.makedirs(val_depth_directory)
	if not os.path.exists(val_annotation_directory):
	    os.makedirs(val_annotation_directory)
	k=1
	printed = True
	temp = '/media/luca/TERZIARIO/ubuntu_data/dataset_luca/XVAC_Dataset/'
	for j in range(acquisition_size):
		temp_rgb = temp
		temp_depth = temp
		temp_annot = temp
		
		temp_rgb+=str(k) + '_grasp/' + str(k) + '_rgb_image.png'
		temp_depth+=str(k) + '_grasp/' + str(k) + '_depth_image.png'
		temp_annot+=str(k) + '_grasp/' + str(k) + '_data.txt'
		
		if j < key2:
			shutil.copyfile(temp_rgb, os.getcwd() + '/' + val_rgb_directory + '/' + str(k) + '.png')	
			shutil.copyfile(temp_depth, os.getcwd() + '/' + val_depth_directory + '/' + str(k) + '.png')
			shutil.copyfile(temp_annot, os.getcwd() + '/' + val_annotation_directory + '/' + str(k) + '.txt')
			#rgb crop process
			img = cv2.imread(os.getcwd() + '/' + val_rgb_directory + '/' + str(k) + '.png')
			crop_img = img[0:424, 0:424]
			cv2.imwrite(os.getcwd() + '/' + val_rgb_directory + '/' + str(k) + '.png', crop_img)
			#depth crop process
			img = cv2.imread(os.getcwd() + '/' + val_depth_directory + '/' + str(k) + '.png',cv2.IMREAD_UNCHANGED)
			crop_img = img[0:424, 0:424]
			cv2.imwrite(os.getcwd() + '/' + val_depth_directory + '/' + str(k) + '.png', crop_img)

			if(float(key2/4) > (j) and printed):
				print("progress [=>    ]20%")
				printed = False
			if(float(key2/3) > (j) and float(key2/(4)) < (j) and (not(printed))):
				print("progress [==>   ]40%")
				printed = True
			if(float(key2/2) > (j) and float(key2/3) < (j) and printed):
				print("progress [===>  ]60%")
				printed = False
			if(float(key2/1.5) > (j) and float(key2/2) < (j) and (not(printed))):
				print("progress [====> ]80%")
				printed = True
		#create training dataset
		else:
			if j == key2:
				k=1
				print("progress [=====>]100%")		
				print("[OK] Validation set succesfully created\n")
				print("Creation of the training set...\n")
			shutil.copyfile(temp_rgb, os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png')	
			shutil.copyfile(temp_depth, os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png')
			shutil.copyfile(temp_annot, os.getcwd() + '/' + train_annotation_directory + '/' + str(k) + '.txt')
			#rgb crop process
			img = cv2.imread(os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png')
			crop_img = img[0:424, 0:424]
			cv2.imwrite(os.getcwd() + '/' + train_rgb_directory + '/' + str(k) + '.png', crop_img)
			#depth crop process
			img = cv2.imread(os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png',cv2.IMREAD_UNCHANGED)
			crop_img = img[0:424, 0:424]
			cv2.imwrite(os.getcwd() + '/' + train_depth_directory + '/' + str(k) + '.png', crop_img)

			if(float((acquisition_size-key2)/4) > (j-key2) and printed):
				print("progress [=>    ]20%")
				printed = False
			if(float((acquisition_size-key2)/3) > (j-key2) and float((acquisition_size-key2)/4) < (j-key2) and (not(printed))):
				print("progress [==>   ]40%")
				printed = True
			if(float((acquisition_size-key2)/2) > (j-key2) and float((acquisition_size-key2)/3) < (j-key2) and printed):
				print("progress [===>  ]60%")
				printed = False
			if(float((acquisition_size-key2)/1.5) > (j-key2) and float((acquisition_size-key2)/2) < (j-key2) and (not(printed))):
				print("progress [====> ]80%")
				printed = True

		k=k+1

	print("progress [=====>]100%")		
	print("[OK] Training set succesfully created")

if(key == 2):
	if not os.path.exists(test_rgb_directory):
		os.makedirs(test_rgb_directory)
	if not os.path.exists(test_depth_directory):
		os.makedirs(test_depth_directory)

	print("Creation of the test image set...\n")
	k=4001
	printed = True
	number_images = 150
	temp = '/media/luca/TERZIARIO/ubuntu_data/dataset_luca/test_image/'
	for j in range(number_images):
		temp_rgb = temp
		temp_depth = temp
		
		temp_rgb+=str(k) + '_grasp/' + str(k) + '_rgb_image.png'
		temp_depth+=str(k) + '_grasp/' + str(k) + '_depth_image.png'
		shutil.copyfile(temp_rgb, os.getcwd() + '/' + test_rgb_directory + '/' + str(k) + '.png')	
		shutil.copyfile(temp_depth, os.getcwd() + '/' + test_depth_directory + '/' + str(k) + '.png')
		#rgb crop process
		img = cv2.imread(os.getcwd() + '/' + test_rgb_directory + '/' + str(k) + '.png')
		crop_img = img[0:424, 0:424]
		cv2.imwrite(os.getcwd() + '/' + test_rgb_directory + '/' + str(k) + '.png', crop_img)
		#depth crop process
		img = cv2.imread(os.getcwd() + '/' + test_depth_directory + '/' + str(k) + '.png',cv2.IMREAD_UNCHANGED)
		crop_img = img[0:424, 0:424]
		cv2.imwrite(os.getcwd() + '/' + test_depth_directory + '/' + str(k) + '.png', crop_img)
		k=k+1
		if(float(number_images/4) > (j) and printed):
			print("progress [=>    ]20%")
			printed = False
		if(float(number_images/3) > (j) and float(number_images/(4)) < (j) and (not(printed))):
			print("progress [==>   ]40%")
			printed = True
		if(float(number_images/2) > (j) and float(number_images/3) < (j) and printed):
			print("progress [===>  ]60%")
			printed = False
		if(float(number_images/1.5) > (j) and float(number_images/2) < (j) and (not(printed))):
			print("progress [====> ]80%")
			printed = True
	print("progress [=====>]100%\n")	
	print("[OK] Succesfully created")
