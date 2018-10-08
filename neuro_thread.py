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
import cv2
import os
from keras.utils.np_utils import to_categorical

np.random.seed(100)

def load_data():
	X_t = []
	Y_t = []
	x = 1
	while x <= ko_num_thread:
		if x <= ok_num_thread:
			img = cv2.imread('data_thread/Ok_'+str(x) + '.jpg')
			X_t.append(img)
			Y_t.append(1)
		img = cv2.imread('data_thread/Ko_'+str(x) + '.jpg')
		X_t.append(img)
		Y_t.append(0)
		x+=1
	# Y_t = to_categorical(Y_t, num_classes=2)
	return train_test_split(X_t, Y_t, test_size = 0.4, random_state = 71)

ko_num_thread =  175
ok_num_thread =  166

# ok_num_thread = raw_input('ok_num_thread: ')
# ko_num_thread = raw_input('ko_num_thread: ')

X_train, X_test, y_train, y_test = load_data()

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

img_rows, img_cols = X_train.shape[1], X_train.shape[2]
input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
X_train /= 255
X_test /= 255
if os.path.exists("./model_weights/model_t.json") and os.path.exists("./model_weights/model_t.h5"):
	print("Load model")
	json_file = open('model_weights/model_t.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model_weights/model_t.h5")
	model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
	
else:
	model = Sequential()

	model.add(Conv2D(30, kernel_size=(5, 5),
			activation='relu',
			input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(30, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(30, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(6, activation='sigmoid'))
	model.add(Dense(3, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
	print(model.summary())

	model.fit(X_train, y_train, epochs=5, validation_split=0.2)

	model_json = model.to_json()
	with open("model_weights/model_t.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model_weights/model_t.h5")
	print("Saved model to disk")


predicted = model.predict(X_test)
alls = len(predicted)
i = 0
while i < alls:
	print predicted[i], ' == ',y_test[i]
	i += 1
print 'min = ', min(predicted)
print 'max = ', max(predicted)

y_test = y_test.astype(float)
fpr, tpr, threshold = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
print 'auc  = ',roc_auc
print 'mean_squared_error = ', mean_squared_error(y_test, predicted)
