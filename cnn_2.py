import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
import matplotlib.pyplot as plt 
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


IMG_SIZE= 50
LR= 1e-3


dataset="E:/Data set for pattern/tr"
testset="E:/Data set for pattern/t"
categories =["non-ped_examples","ped_examples"]
training_data=[] 
test_data=[] 
X_train=[]
X_test=[]
IMG_SIZE = 50
MODEL_NAME= 'pedestrain problem'
#E:/Data set for pattern/t/non-ped_examples
def create_training_data(training_data,dataset,cat,IMG_SIZE):
    for category in cat:
        path = os.path.join(dataset,category)
        class_num = categories.index(category)
        if class_num == 0 :
            class_num = np.array([1,0])
        elif class_num == 1 :
            class_num = np.array([0,1])
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
            training_data.append([np.array(new_array),class_num])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
            


def create_test_data(test_data,testset,cat,IMG_SIZE):
    for category in cat:
        path = os.path.join(testset,category)
        class_num = categories.index(category)
        if class_num == 0 :
            class_num = np.array([1,0])
        elif class_num == 1 :
            class_num = np.array([0,1])
        for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                test_data.append([np.array(new_array),class_num])
    shuffle(test_data)
    np.save('test_data.npy', test_data)


def formalize(X,y,data):
    for features , labels in data:
        X.append(features)
        y.append(labels)
#Data set and Test set Loading        
create_training_data(training_data,dataset,categories,IMG_SIZE)
create_test_data(test_data,testset,categories,IMG_SIZE)
#formalize(X_train,y_train,training_data)
#formalize(X_test,y_test,test_data)
X_train=np.array([i[0] for i in training_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train=[i[1]for i in training_data]
X_test=np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_test=[i[1]for i in test_data]

#Building the model

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet=conv_2d(convnet, 32, 5, activation= 'relu')
convnet= max_pool_2d(convnet, 5)
convnet=conv_2d(convnet, 64, 5, activation= 'relu')
convnet= max_pool_2d(convnet, 5)
convnet=conv_2d(convnet, 128, 5, activation= 'relu')
convnet= max_pool_2d(convnet, 5)
convnet=conv_2d(convnet, 64, 5, activation= 'relu')
convnet= max_pool_2d(convnet, 5)
convnet=conv_2d(convnet, 32, 5, activation= 'relu')
convnet= max_pool_2d(convnet, 5)
convnet=fully_connected(convnet, 1024, activation= 'relu')
convnet= dropout(convnet, 0.8)
convnet=fully_connected(convnet, 2, activation= 'softmax')
convnet= regression(convnet,optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets' )
model= tflearn.DNN(convnet, tensorboard_dir= 'log', tensorboard_verbose=0)
model.fit( X_train, y_train, n_epoch=10,validation_set=({'input': X_test}, {'targets': y_test}),snapshot_step= 500, show_metric= True, run_id= MODEL_NAME)
#model.fit({'input': X_train},{'targets': y_train}, n_epoch=10,validation_set=({'input': X_test}, {'targets': y_test}),snapshot_step= 500, show_metric= True, run_id= MODEL_NAME)
fig= plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):
	img_num= data[1]
	img_data= data[0]
	y= fig.add_subplot(4, 4, num+1)
	orig= img_data
	data= img_data.reshape(IMG_SIZE,IMG_SIZE,1)
	model_out= model.predict([data])[0]
	if np.argmax(model_out)== 1:
		str_label= 'pedestrain'
	else:
		str_label= 'non-pedestrain'
	y.imshow(orig, cmap= 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()
