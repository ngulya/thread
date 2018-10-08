#!/usr/bin/env python
import cv2 as cv2
import sys
import numpy as np
import time as t
from time import sleep

def f_i_old(x):
	global i_old
	i_old = x

def f_x_old(x):
	global x_old
	x_old = x

def f_y_old(x):
	global y_old
	y_old = x


def draw_rectangle(result, x, y):
	cv2.rectangle(result,(x,y),(x + 310,y + 560),(255,255,255),2)
	cv2.line(result,(x + 70 + 50,y),(x + 70 + 50,y + 560),(255,255,255),2)
	cv2.line(result,(x + 140 + 50,y),(x + 140 + 50,y + 560),(255,255,255),2)

file_name = 'Test_01.mp4'

cv2.namedWindow('frame')
cv2.moveWindow('frame',250,150)

try:
	cap = cv2.VideoCapture(file_name)
	tots = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	cv2.createTrackbar('S','frame', 0,int(tots)-1, f_i_old)
	cv2.setTrackbarPos('S','frame',0)
except Exception as e:
	print 'No video Test_01.mp4'
	exit()


x_old = 0
y_old = 0

x = 0
y = 0

x1 = 0
x2 = 0
x3 = 0
x4 = 0

y1 = 0
y2 = 0
y3 = 0
y4 = 0

i = 0
i_old = 0

frame_rate = 24

space = False
x1 = 588##
y1 = 150##

must = float(1)/frame_rate
ko_num_thread = 0
ok_num_thread = 0
ko_num_l_r = 0
ok_num_l_r = 0
while True:
	c1 = t.time()
	if space == False:
		flag, frame = cap.read()
	if flag:
		# canny = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		# canny = cv2.Canny(frame,30,50)
		
		if x1 != 0 and y1 != 0:
			draw_rectangle(frame, x1, y1)
		if x2 != 0 and y2 != 0:
			draw_rectangle(frame, x2, y2)
		if x3 != 0 and y3 != 0:
			draw_rectangle(frame, x3, y3)
		if x4 != 0 and y4 != 0:
			draw_rectangle(frame, x4, y4)
		cv2.imshow('frame',frame)
		

	k = cv2.waitKey(1)
	if x_old != x or y_old != y:
		x = x_old
		y = y_old
	if i_old != i:
		print i_old, i
		if i_old == -1:
			break
		cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i_old)
		i = i_old

	
	if k == 1048603:#esc
		break
	elif k == 1048608:#space
		if space == True:
			space = False
		else:
			space = True
	
	
	if (10 < i < 1140) or (1250 < i < 2960) or (3200 < i < 5400):
		if i % 40 == 0:
			ok_num_l_r += 1
			tmp = frame[y1:y1+560]
			tmp = np.concatenate((tmp[:, x1:x1 + 70 + 50], tmp[:, x1 + 140 + 50:x1 + 210 + 50 +50]), axis = 1)
			cv2.imwrite("./data_l_r/"+"Ok_"+str(ok_num_l_r)+".jpg",tmp)
	elif (1145 < i < 1245) or (2970 < i < 3018):
		ko_num_l_r += 1
		tmp = frame[y1:y1+560]
		tmp = np.concatenate((tmp[:, x1:x1 + 70 + 50], tmp[:, x1 + 140 + 50:x1 + 210 + 50 +50]), axis = 1)
		cv2.imwrite("./data_l_r/"+"Ko_"+str(ko_num_l_r)+".jpg",tmp)

	#save thread
	if (10 < i < 1140) or (1250 < i < 2960) or (3120 < i < 3640):#3360
		if i % 20 == 0:
			ok_num_thread += 1
			tmp = frame[y1:y1+560, x1 + 70 + 50:x1 + 140 + 50]
			cv2.imwrite("./data_thread/"+"Ok_"+str(ok_num_thread)+".jpg",tmp)
	elif (3640 < i < 5400):#1760
		if i % 10 == 0:
			ko_num_thread += 1
			tmp = frame[y1:y1+560, x1 + 70 + 50:x1 + 140 + 50]
			cv2.imwrite("./data_thread/"+"Ko_"+str(ko_num_thread)+".jpg",tmp)
	if i > 5400:
		break
	

	
	# c2 = t.time()
	# kof = must - (c2-c1)
	# if kof > 0:
	# 	sleep(kof)
	if space == False: i += 1
	cv2.setTrackbarPos('S','frame',i)

if ko_num_thread != 175:
	print 'Change in neuro_thread.py ko_num_thread = ', ko_num_thread
if ok_num_thread != 166:
	print 'Change in neuro_thread.py ok_num_thread = ', ok_num_thread

if ko_num_l_r != 146:
	print 'Change in neuro_l_r.py ko_num_l_r = ', ko_num_l_r
if ok_num_l_r != 124:
	print 'Change in neuro_l_r.py ok_num_l_r = ', ok_num_l_r