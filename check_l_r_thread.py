#!/usr/bin/env python
from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc
import numpy as np

import os
from keras.utils.np_utils import to_categorical
import cv2 as cv2
import sys
import time as t
from time import sleep
np.random.seed(100)


def load_data():
	global model, minn, maxx
	global model_t, minn_t, maxx_t

	print 'First time need load models '
	if os.path.exists("./model_weights/model_l_r.json") and os.path.exists("./model_weights/model_l_r.h5"):	
		json_file = open('model_weights/model_l_r.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights("model_weights/model_l_r.h5")
		model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
		minn = 0.47340256
		maxx = 0.50628877
	else:
		print 'Error: No model for left_right'
		exit(1)

	if os.path.exists("./model_weights/model_t.json") and os.path.exists("./model_weights/model_t.h5"):
		json_file = open('model_weights/model_t.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model_t = model_from_json(loaded_model_json)
		model_t.load_weights("model_weights/model_t.h5")
		model_t.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
		minn_t = 0.35472813
		maxx_t = 0.45125076
	else:
		print 'Error: No model for thread'
		exit()


def predicted_l_r(Img):
	global minn, maxx, model
	if minn == 0 and maxx == 0:
		load_data()
	Img = np.asarray(Img)
	Img = Img.astype(float)
	Img /= 255
	
	predicted = model.predict(Img)[0][0]
	probc = ((predicted - minn) / (maxx - minn))*100
	return probc

def predicted_thread(img):
	global minn_t, maxx_t, model_t

	img = np.asarray(img)
	img = img.astype(float)
	img /= 255
	
	predicted = model_t.predict(img)[0][0]
	probt = ((predicted - minn_t) / (maxx_t - minn_t))*100
	return probt



def f_i_old(x):
	global i_old
	i_old = x

def draw_rectangle(result, x, y):
	cv2.rectangle(result,(x,y),(x + 310,y + 560),(255,255,255),2)
	cv2.line(result,(x + 70 + 50,y),(x + 70 + 50,y + 560),(255,255,255),2)
	cv2.line(result,(x + 140 + 50,y),(x + 140 + 50,y + 560),(255,255,255),2)


def return_x_y(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		global x1, y1, i
		if i > 5400:
			x1 = x
			y1 = y


file_name = 'Test_01.mp4'


cv2.namedWindow('frame')
cv2.moveWindow('frame',250,150)

cap = cv2.VideoCapture(file_name)

tots = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

cv2.createTrackbar('S','frame', 0,int(tots)-1, f_i_old)
cv2.setTrackbarPos('S','frame',0)


cv2.setMouseCallback('frame',return_x_y)

model = 0
minn = 0
maxx = 0

model_t = 0
minn_t = 0
maxx_t = 0


x_old = 0
y_old = 0

x = 0
y = 0

x1 = 0
y1 = 0

i = 0
i_old = 0

times = 1
frame_rate = 24

space = False
x1 = 588##
y1 = 150##

print '\n\nPress p for detecting\nAfter 5400 frame can change rectangle\n'
must = float(1)/frame_rate
oks_num = 0
koks_num = 0
while True:
	c1 = t.time()
	if space == False:
		flag, frame = cap.read()
		# frame = frame[149:149+660, 593:593 + 400]
	if flag:
		result = frame.copy()

		if x1 != 0 and y1 != 0:
			draw_rectangle(result, x1, y1)
		cv2.imshow('frame',result)
		

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
	elif k == 1048625:#1
		times = 1
		x1 = 0
		y1 = 0
	elif k == 1048688:#p
		vuvod = frame[y1:y1+560, x1:x1+310]
		cv2.line(vuvod,(70 + 50,0),(70 + 50,560),(12,12,230),2)
		cv2.line(vuvod,(140 + 50,0),(140 + 50,560),(12,12,230),2)
		cv2.imshow('input in nn',vuvod)
		tmp = frame[y1:y1+560]
		tmp = np.concatenate((tmp[:, x1:x1 + 70 + 50], tmp[:, x1 + 140 + 50:x1 + 210 + 50 +50]), axis = 1)	
		probc = predicted_l_r([tmp])
		print 'All Clear = %.4f%%'%(probc)
		if probc > 50:
			tmp2 = frame[y1:y1+560, x1 + 70 + 50:x1 + 140 + 50]
			probt = predicted_thread([tmp2])
			print 'Prob. have thread = %.4f%%'%(probt)
			if probt < 50:
				print 'Alarm: No thread'
			print
		k = cv2.waitKey(1)
	elif k != -1:
		print 'you press -', k
		print 'if this p, need change 183 line'

	c2 = t.time()
	kof = must - (c2-c1)
	if kof > 0:
		sleep(kof)
	if space == False: i += 1
	cv2.setTrackbarPos('S','frame',i)
