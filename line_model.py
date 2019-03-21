from __future__ import absolute_import, division, print_function
import tensorflow as tf
from api import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import cv2
import os

#load and verify the data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

def eval_class(data):
	length = data[0]
	width = data[1] 	
	angle = data[2]
	colour = data[3]
	base = 0
	if length=='0':
		if width == '0':
			if colour == '0':
				class_id = int(angle)
			else:
				class_id = 12 + int(angle)
		else:
			if colour == '0':
				class_id = 24+int(angle)
			else:
				class_id = 36+ int(angle)
	else:
		if width == '0':
			if colour == '0':
				class_id = 48+int(angle)
			else:
				class_id = 60 + int(angle)
		else:
			if colour == '0':
				class_id = 72+int(angle)
			else:
				class_id = 84+ int(angle)
	return class_id


images = os.listdir("data/")
X = []
Y = []
for image in images:
	img=cv2.imread("data/"+image)
	X.append(img)
	data=image[:-4].split('_')
	Y.append(eval_class(data))
	# print(image)

X=np.asarray(X)
Y=np.asarray(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]*3)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]*3)
print("Training Dataset_shape",x_train.shape,y_train.shape)
print("Testing Dataset_shape",x_test.shape,y_test.shape)

x_train=x_train/255.0
x_test=x_test/255.0

# #create the model
model = layers()
model.Input(28,84)
model.add_layer(128,"relu")
model.add_layer(96,"softmax")
model.compile()

# define useful training variables
epochs =5
batch_size = 100

#start the training
model.train(x_train,y_train,batch_size,epochs)
print("Training Complete")

#Evaluate on the test_set
model.predict(x_test, y_test)

#get the confusion matrix
model.confusion(y_test)
# print(x_train.shape)
# get the learning curves
# estimator = GaussianNB()
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# title = "Learning Curve"
# x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# model.learning_curve(estimator,title,x_train[:5000],y_train[:5000],cv=cv)
# plt.show()