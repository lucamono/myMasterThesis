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

def grids_batch_generator(YVAC_data,flag,IMAGE_H, IMAGE_W, GRID_H, GRID_W, BATCH_IMG_TRAIN_SIZE, BATCH_IMG_VAL_SIZE):
	print("Prepare the batch for the Network...")
	GRID_W_px = (IMAGE_W / GRID_W)
	GRID_H_px = (IMAGE_H / GRID_H)

	#case of training data
	if(flag==0):
		batch_size = BATCH_IMG_TRAIN_SIZE
		#batch_size = 284
	#case of test data
	if(flag==1):
		batch_size = BATCH_IMG_VAL_SIZE
	y_batch = np.zeros((batch_size, GRID_H, GRID_W, 5)) 

	for j in range(batch_size): 

		groun_truth_x = YVAC_data[j][0]
		groun_truth_y = YVAC_data[j][1]
		pitch = YVAC_data[j][2]
		roll = YVAC_data[j][3]
		grasp = float(YVAC_data[j][4])
		pos_x_tmp = groun_truth_x / (float(IMAGE_W) / GRID_W)
		pos_y_tmp = groun_truth_y / (float(IMAGE_H) / GRID_H)
		grid_x = int(np.floor(pos_x_tmp))
		grid_y = int(np.floor(pos_y_tmp))

		GRID_W_px = (IMAGE_W / GRID_W)
		GRID_H_px = (IMAGE_H / GRID_H)

		if(grid_x*GRID_W_px < groun_truth_x):
			shift_x = (groun_truth_x - grid_x*GRID_W_px)/ (2.0*float(GRID_W_px))
		else:
			shift_x = (grid_x*GRID_W_px - groun_truth_x)/ (2.0*float(GRID_W_px))

		if(grid_y*GRID_H_px < groun_truth_y):
			shift_y = (groun_truth_y - grid_y*GRID_H_px)/ (2.0*float(GRID_W_px))
		else:
			shift_y = (grid_y*GRID_H_px - groun_truth_y)/ (2.0*float(GRID_W_px))
		
		shift_x = shift_x + 0.5
		shift_y = shift_y + 0.5

		#set the groundtruth values of the cell that contains the grasping point
		y_batch[j, grid_y, grid_x, 0] = shift_x 
		y_batch[j, grid_y, grid_x, 1] = shift_y 
		y_batch[j, grid_y, grid_x, 2] = XVAC2Numpy.normalizeAnglePitch(pitch)
		y_batch[j, grid_y, grid_x, 3] = XVAC2Numpy.normalizeAngleRoll(roll)
		y_batch[j, grid_y, grid_x, 4] = grasp


		#the list of the nearest cells
		list_nearest_cells = []
		graspOrNot = grasp

		cell = [grid_x, grid_y] 
		list_nearest_cells.append(cell)

		#the right nearest cells condition
		if(int((shift_x-0.5)*(2.0*float(GRID_W_px))) > (GRID_W_px/2.0)):
			flag_cell = 1
		#else the condition is the left nearest cells
		else:
			flag_cell = -1
		
		if(((grid_x+flag_cell) < GRID_W) and ((grid_x+flag_cell) > 0)):
			cell = [(grid_x+flag_cell), grid_y] 
			list_nearest_cells.append(cell)
			
		
		#the bottom nearest cell condition
		if(int((shift_y-0.5)*(2.0*float(GRID_H_px))) > (GRID_H_px/2.0)):
			#check boundary grid image condition
			if((grid_x+flag_cell) < GRID_W and ((grid_x+flag_cell) > 0) and (grid_y+1) < GRID_H and (grid_y+1) > 0): 
				cell = [(grid_x+flag_cell), (grid_y+1)] 
				list_nearest_cells.append(cell)

			if((grid_y+1) < GRID_H and (grid_y+1) > 0):
				cell2 = [grid_x, (grid_y+1)]
				list_nearest_cells.append(cell2)
		#the top nearest cell condition
		else:
			if(((grid_x+flag_cell) < GRID_W) and ((grid_x+flag_cell) > 0) and (grid_y - 1) < GRID_H and  ((grid_y - 1) > 0) ):
				cell = [(grid_x+flag_cell), (grid_y-1)]
				list_nearest_cells.append(cell)
			if((grid_y - 1) < GRID_H and  ((grid_y - 1) > 0)):
				cell2 = [grid_x, (grid_y-1)]
				list_nearest_cells.append(cell2)
		
		cell1_x = list_nearest_cells[0][0]
		cell1_y = list_nearest_cells[0][1]
		draw1_x = int(cell1_x*int(GRID_H_px) + int((shift_x-0.5)*(2.0*float(GRID_W_px))))
		draw1_y = int(cell1_y*int(GRID_H_px)+int((shift_y-0.5)*(2.0*float(GRID_W_px))))

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

		#set the groundtruth values of the nearest cells of the grasping point
		y_batch[j, cell2_y, cell2_x, 0] = shift2_x 
		y_batch[j, cell2_y, cell2_x, 1] = shift2_y 
		y_batch[j, cell2_y, cell2_x, 2] = XVAC2Numpy.normalizeAnglePitch(pitch)
		y_batch[j, cell2_y, cell2_x, 3] = XVAC2Numpy.normalizeAngleRoll(roll)
		y_batch[j, cell2_y, cell2_x, 4] = grasp

		y_batch[j, cell3_y, cell3_x, 0] = shift3_x 
		y_batch[j, cell3_y, cell3_x, 1] = shift3_y 
		y_batch[j, cell3_y, cell3_x, 2] = XVAC2Numpy.normalizeAnglePitch(pitch)
		y_batch[j, cell3_y, cell3_x, 3] = XVAC2Numpy.normalizeAngleRoll(roll)
		y_batch[j, cell3_y, cell3_x, 4] = grasp

		y_batch[j, cell4_y, cell4_x, 0] = shift4_x 
		y_batch[j, cell4_y, cell4_x, 1] = shift4_y 
		y_batch[j, cell4_y, cell4_x, 2] = XVAC2Numpy.normalizeAnglePitch(pitch)
		y_batch[j, cell4_y, cell4_x, 3] = XVAC2Numpy.normalizeAngleRoll(roll)
		y_batch[j, cell4_y, cell4_x, 4] = grasp

	return y_batch	

