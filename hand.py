# -*- coding: utf-8 -*-
#data.encoding='gb2312'
"""
Created on Sat Dec 21 10:32:09 2019

@author: 26330
"""
#from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers,Sequential,datasets,losses,optimizers
from tensorflow.keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout,Conv2D

#导入数据
(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data()
 
#标准化
im_row=28
im_col=28

x_train=x_train.reshape(x_train.shape[0],1,im_row,im_col)
x_test=x_test.reshape(x_test.shape[0],1,im_row,im_col)
x_train=x_train/255
x_test=x_test/255
#
# N1=4
# for i in range(N1*N1):
#     image=x_train[i][0]
#     plt.subplot(N1,N1,i+1)
#     plt.grid(False)
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(image, cmap=plt.cm.binary)
# plt.show()

#进行热编码
batch_size = 64
num_classes = 10
epochs = 12
y_train=tf.one_hot(y_train,depth=10)
y_test=tf.one_hot(y_test,depth=10)

x_train=x_train.reshape(x_train.shape[0],im_row,im_col,1)
x_test=x_test.reshape(x_test.shape[0],im_row,im_col,1)
print(y_train.shape)
print(x_train.shape)



model = Sequential()   #更正：不是model.Sequential()
input_shape = [im_row,im_col,1]
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape = input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))   #更正：这里是pool_size 不是kernel_size 
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))


model.compile(loss =losses.categorical_crossentropy,optimizer =optimizers.Adadelta(),metrics=['accuracy'])  #这里的optimizers 有s
history=model.fit(x_train,y_train,batch_size = batch_size ,epochs = epochs,verbose=2,validation_data=(x_test,y_test))  #注意是epochs 不是es

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()






















