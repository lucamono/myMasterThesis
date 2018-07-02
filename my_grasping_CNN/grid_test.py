import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2 
import numpy as np
import XVAC2Numpy

def my_min(sequence):
    """return the minimum element of sequence"""
    low = sequence[0] # need to start with some value
    for i in sequence:
        if i < low:
            low = i
    return low

def my_max(sequence):
    """return the maximum element of sequence"""
    max = sequence[0] # need to start with some value
    for i in sequence:
        if i > max:
            max = i
    return max

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

    GRID_W_px = (IMAGE_W / GRID_W)
    GRID_H_px = (IMAGE_H / GRID_H)
       
    if(grid_x*GRID_W_px < groun_truth_x):
        shift_x = (groun_truth_x - grid_x*GRID_W_px)/(2.0*float(GRID_W_px))
    else:
        shift_x = (grid_x*GRID_W_px - groun_truth_x)/(2.0*float(GRID_W_px))

    if(grid_y*GRID_H_px < groun_truth_y):
        shift_y = (groun_truth_y - grid_y*GRID_H_px)/(2.0*float(GRID_W_px))
    else:
        shift_y = (grid_y*GRID_H_px - groun_truth_y)/(2.0*float(GRID_W_px))

    y_batch[0, grid_y, grid_x, 0] = shift_x + 0.5
    y_batch[0, grid_y, grid_x, 1] = shift_y + 0.5
    y_batch[0, grid_y, grid_x, 2] = XVAC2Numpy.normalizeAngleRoll(roll)
    y_batch[0, grid_y, grid_x, 3] = XVAC2Numpy.normalizeAnglePitch(pitch)
    y_batch[0, grid_y, grid_x, 4] = grasp

    return (y_batch, grid_x, grid_y, groun_truth_x, groun_truth_y)

IMAGE_H, IMAGE_W = 424, 424
GRID_H,  GRID_W  = 8 , 8
BATCH_SIZE = 2403

GRID_W_px = (IMAGE_W / GRID_W)
GRID_H_px = (IMAGE_H / GRID_H)

grid_img_path = '/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/temp/grid_test.png'
grid_img = cv2.imread (grid_img_path)

#draw grids on grid_image
for i in range(GRID_W):
	for j in range(GRID_H):
		cv2.rectangle(grid_img, (i*int(GRID_H_px), j*int(GRID_H_px)), (i*int(GRID_H_px)+int(GRID_H_px), j*int(GRID_H_px)+int(GRID_H_px)), (0,0,0), 1)	

#load GroundTruth INDEX
temp_img = '/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/XVAC2NETWORK/train/rgb_images/'
temp_annotation = '/media/luca/TERZIARIO/ubuntu_data/my_grasping_CNN/XVAC2NETWORK/train/annotations/'
k = 1

#the list of the nearest cells

for i in range(BATCH_SIZE):
	list_nearest_cells = []
	#temp grid image for each image in dataset
	grid_img_clone = grid_img.copy()
	
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

	#draw the current groundtruth cell in the rgb image
	cv2.rectangle(image, (grid_x*int(GRID_H_px), grid_y*int(GRID_H_px)), (grid_x*int(GRID_H_px)+int(GRID_H_px), grid_y*int(GRID_H_px)+int(GRID_H_px)), (0,255,0), 1)
	#draw the current groundtruth point of grasping in the rgb image
	cv2.circle(image,(grid_x*int(GRID_H_px) + int((shift_x-0.5)*(2.0*float(GRID_W_px))), grid_y*int(GRID_H_px)+int((shift_y-0.5)*(2.0*float(GRID_W_px)))), 4, (0,255,0), -1)
	
	#draw and store the cell that contains the point
	cv2.rectangle(grid_img_clone, (grid_x*int(GRID_H_px), grid_y*int(GRID_H_px)), (grid_x*int(GRID_H_px)+int(GRID_H_px), grid_y*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)
	cell = [grid_x, grid_y] 
	list_nearest_cells.append(cell)

	#the right nearest cells condition
	if(int((shift_x-0.5)*(2.0*float(GRID_W_px))) > (GRID_W_px/2.0)):
		flag_cell = 1
	#else the condition is the left nearest cells
	else:
		flag_cell = -1
	
	if(((grid_x+flag_cell) < GRID_W) and ((grid_x+flag_cell) > 0)):
		cv2.rectangle(grid_img_clone, ((grid_x+flag_cell)*int(GRID_H_px), grid_y*int(GRID_H_px)), ((grid_x+flag_cell)*int(GRID_H_px)+int(GRID_H_px), grid_y*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)
		cell = [(grid_x+flag_cell), grid_y] 
		list_nearest_cells.append(cell)
		
	
	#the bottom nearest cell condition
	if(int((shift_y-0.5)*(2.0*float(GRID_H_px))) > (GRID_H_px/2.0)):
		#check boundary grid image condition
		if((grid_x+flag_cell) < GRID_W and ((grid_x+flag_cell) > 0) and (grid_y+1) < GRID_H and (grid_y+1) > 0): 
			cv2.rectangle(grid_img_clone, ((grid_x+flag_cell)*int(GRID_H_px), (grid_y+1)*int(GRID_H_px)), ((grid_x+flag_cell)*int(GRID_H_px)+int(GRID_H_px), (grid_y+1)*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)
			cell = [(grid_x+flag_cell), (grid_y+1)] 
			list_nearest_cells.append(cell)

		if((grid_y+1) < GRID_H and (grid_y+1) > 0):
			cv2.rectangle(grid_img_clone, ((grid_x)*int(GRID_H_px), (grid_y+1)*int(GRID_H_px)), ((grid_x)*int(GRID_H_px)+int(GRID_H_px), (grid_y+1)*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)
			cell2 = [grid_x, (grid_y+1)]
			list_nearest_cells.append(cell2)
	#the top nearest cell condition
	else:
		if(((grid_x+flag_cell) < GRID_W) and ((grid_x+flag_cell) > 0) and (grid_y - 1) < GRID_H and  ((grid_y - 1) > 0) ):
			cv2.rectangle(grid_img_clone, ((grid_x+flag_cell)*int(GRID_H_px), (grid_y-1)*int(GRID_H_px)), ((grid_x+flag_cell)*int(GRID_H_px)+int(GRID_H_px), (grid_y-1)*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)
			cell = [(grid_x+flag_cell), (grid_y-1)]
			list_nearest_cells.append(cell)
		if((grid_y - 1) < GRID_H and  ((grid_y - 1) > 0)):
			cv2.rectangle(grid_img_clone, ((grid_x)*int(GRID_H_px), (grid_y-1)*int(GRID_H_px)), ((grid_x)*int(GRID_H_px)+int(GRID_H_px), (grid_y-1)*int(GRID_H_px)+int(GRID_H_px)), (0,0,255), 2)	
			cell2 = [grid_x, (grid_y-1)]
			list_nearest_cells.append(cell2)
	
	#draw the current groundtruth point of grasping in the grid image
	cv2.circle(grid_img_clone,(grid_x*int(GRID_H_px) + int((shift_x-0.5)*(2.0*float(GRID_W_px))), grid_y*int(GRID_H_px)+int((shift_y-0.5)*(2.0*float(GRID_W_px)))), 4, (0,0,0), -1)

	#Displayed the image
	cv2.imshow("Grasped Points", image)

	grid_img_cell1 = grid_img_clone.copy()
	grid_img_cell2 = grid_img_clone.copy()
	grid_img_cell3 = grid_img_clone.copy()
	grid_img_cell4 = grid_img_clone.copy()

	cell1_x = list_nearest_cells[0][0]
	cell1_y = list_nearest_cells[0][1]
	draw1_x = int(cell1_x*int(GRID_H_px) + int((shift_x-0.5)*(2.0*float(GRID_W_px))))
	draw1_y = int(cell1_y*int(GRID_H_px)+int((shift_y-0.5)*(2.0*float(GRID_W_px))))
	cv2.rectangle(grid_img_cell1, (cell1_x*int(GRID_H_px), cell1_y*int(GRID_H_px)), (cell1_x*int(GRID_H_px)+int(GRID_H_px), cell1_y*int(GRID_H_px)+int(GRID_H_px)), (0,255,0), 2)
	cv2.circle(grid_img_cell1,(draw1_x, draw1_y), 4, (0,255,0), -1)

	case_sign_box = 0

	list_sign_x = []
	list_sign_y = []
	for i in range(len(list_nearest_cells)):
		list_sign_x.append(list_nearest_cells[i][0])
		list_sign_y.append(list_nearest_cells[i][1])
	min_x = my_min(list_sign_x)
	min_y = my_min(list_sign_y)
	max_x = my_max(list_sign_x)
	max_y = my_max(list_sign_y)
	
	#case for box signs
	if(min_x == cell1_x and min_y == cell1_y):
		case_sign_box = 0
	if(min_x == cell1_x and max_y == cell1_y):
		case_sign_box = 1
	if(max_x == cell1_x and min_y == cell1_y):
		case_sign_box = 2
	if(max_x == cell1_x and max_y == cell1_y):
		case_sign_box = 3

	shift2_x = 0
	shift2_y = 0
	shift3_x = 0
	shift3_y = 0
	shift4_x = 0
	shift4_y = 0

	cell2_x = list_nearest_cells[1][0]
	cell2_y = list_nearest_cells[1][1]
	cell3_x = list_nearest_cells[2][0]
	cell3_y = list_nearest_cells[2][1]
	cell4_x = list_nearest_cells[3][0]
	cell4_y = list_nearest_cells[3][1]

	if(case_sign_box == 0):
		if(cell2_x > cell1_x):
			shift2_x = shift_x - 0.5
		else:
			shift2_x = shift_x
		if(cell2_y > cell1_y):
			shift2_y = shift_y - 0.5 
		else:
			shift2_y = shift_y

		if(cell3_x > cell1_x):
			shift3_x = shift_x - 0.5
		else:
			shift3_x = shift_x
		if(cell3_y > cell1_y):
			shift3_y = shift_y - 0.5 
		else:
			shift3_y = shift_y

		if(cell4_x > cell1_x):
			shift4_x = shift_x - 0.5
		else:
			shift4_x = shift_x
		if(cell4_y > cell1_y):
			shift4_y = shift_y - 0.5 
		else:
			shift4_y = shift_y


	if(case_sign_box == 1):
		if(cell2_x > cell1_x):
			shift2_x = shift_x - 0.5
		else:
			shift2_x = shift_x
		if(cell2_y > cell1_y):
			shift2_y = shift_y - 0.5 
		else:
			shift2_y = shift_y

		if(cell3_x > cell1_x):
			shift3_x = shift_x - 0.5
		else:
			shift3_x = shift_x
		if(cell3_y > cell1_y):
			shift3_y = shift_y 
		else:
			shift3_y = shift_y + 0.5

		if(cell4_x > cell1_x):
			shift4_x = shift_x - 0.5
		else:
			shift4_x = shift_x
		if(cell4_y > cell1_y):
			shift4_y = shift_y 
		else:
			shift4_y = shift_y + 0.5 


	if(case_sign_box == 2):
		if(cell2_x > cell1_x):
			shift2_x = shift_x 
		else:
			shift2_x = shift_x + 0.5
		if(cell2_y > cell1_y):
			shift2_y = shift_y + 0.5 
		else:
			shift2_y = shift_y

		if(cell3_x > cell1_x):
			shift3_x = shift_x 
		else:
			shift3_x = shift_x + 0.5
		if(cell3_y > cell1_y):
			shift3_y = shift_y - 0.5
		else:
			shift3_y = shift_y 

		if(cell4_x > cell1_x):
			shift4_x = shift_x + 0.5
		else:
			shift4_x = shift_x

		if(cell4_y > cell1_y):
			shift4_y = shift_y - 0.5 
		else:
			shift4_y = shift_y 

	if(case_sign_box == 3):
		if(cell2_x > cell1_x):
			shift2_x = shift_x 
		else:
			shift2_x = shift_x + 0.5
		if(cell2_y > cell1_y):
			shift2_y = shift_y + 0.5 
		else:
			shift2_y = shift_y

		if(cell3_x > cell1_x):
			shift3_x = shift_x 
		else:
			shift3_x = shift_x + 0.5
		if(cell3_y > cell1_y):
			shift3_y = shift_y 
		else:
			shift3_y = shift_y + 0.5

		if(cell4_x > cell1_x):
			shift4_x = shift_x + 0.5
		else:
			shift4_x = shift_x

		if(cell4_y > cell1_y):
			shift4_y = shift_y  
		else:
			shift4_y = shift_y + 0.5

	draw2_x = int(cell2_x*int(GRID_H_px) + int((shift2_x-0.5)*(2.0*float(GRID_W_px))))
	draw2_y = int(cell2_y*int(GRID_H_px) + int((shift2_y-0.5)*(2.0*float(GRID_W_px))))
	cv2.rectangle(grid_img_cell2, (cell2_x*int(GRID_H_px), cell2_y*int(GRID_H_px)), (cell2_x*int(GRID_H_px)+int(GRID_H_px), cell2_y*int(GRID_H_px)+int(GRID_H_px)), (0,255,0), 2)
	cv2.circle(grid_img_cell2,(draw2_x, draw2_y), 4, (0,255,0), -1)

	draw3_x = int(cell3_x*int(GRID_H_px) + int((shift3_x-0.5)*(2.0*float(GRID_W_px))))
	draw3_y = int(cell3_y*int(GRID_H_px) + int((shift3_y-0.5)*(2.0*float(GRID_W_px))))
	cv2.rectangle(grid_img_cell3, (cell3_x*int(GRID_H_px), cell3_y*int(GRID_H_px)), (cell3_x*int(GRID_H_px)+int(GRID_H_px), cell3_y*int(GRID_H_px)+int(GRID_H_px)), (0,255,0), 2)
	cv2.circle(grid_img_cell3,(draw3_x, draw3_y), 4, (0,255,0), -1)

	draw4_x = int(cell4_x*int(GRID_H_px) + int((shift4_x-0.5)*(2.0*float(GRID_W_px))))
	draw4_y = int(cell4_y*int(GRID_H_px) + int((shift4_y-0.5)*(2.0*float(GRID_W_px))))
	cv2.rectangle(grid_img_cell4, (cell4_x*int(GRID_H_px), cell4_y*int(GRID_H_px)), (cell4_x*int(GRID_H_px)+int(GRID_H_px), cell4_y*int(GRID_H_px)+int(GRID_H_px)), (0,255,0), 2)
	cv2.circle(grid_img_cell4,(draw4_x, draw4_y), 4, (0,255,0), -1)
	
	cv2.imshow("Test grid cell 1", grid_img_cell1)
	cv2.imshow("Test grid cell 2", grid_img_cell2)
	cv2.imshow("Test grid cell 3", grid_img_cell3)
	cv2.imshow("Test grid cell 4", grid_img_cell4)

	cv2.waitKey(0)
	k = k+1	

