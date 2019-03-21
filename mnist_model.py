#import necessary libraries
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from api import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.naive_bayes import GaussianNB

#load and verify the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training Dataset_shape",x_train.shape,y_train.shape)
print("Testing Dataset_shape",x_test.shape,y_test.shape)

#specify the class names
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#normalize input data
x_train=x_train/255.0
x_test=x_test/255.0

#create the model
model = layers()
model.Input(28,28)
model.add_layer(128,"relu")
model.add_layer(10,"softmax")
model.compile()

#define useful training variables
epochs =5
batch_size = 16

#start the training
model.train(x_train,y_train,batch_size,epochs)
print("Training Complete")

#Evaluate on the test_set
model.predict(x_test, y_test)

#get the confusion matrix
model.confusion(y_test)

#get the learning curves
estimator = GaussianNB()
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
title = "Learning Curve"
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
model.learning_curve(estimator,title,x_train[:500],y_train[:500],cv=cv)
plt.show()	