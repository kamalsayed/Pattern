import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
import cv2
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
dataset="E:/Data set for pattern/2"
testset="E:/Data set for pattern/T2"
categories =["non-ped_examples","ped_examples"]
training_data=[] 
test_data=[] 
Xtr=[]
ytr=[]
Xt=[]
yt=[]
IMG_SIZE = 50
k = 10
#E:/Data set for pattern/t/non-ped_examples
def create_training_data(training_data,dataset,cat,IMG_SIZE):
    for category in cat:
        path = os.path.join(dataset,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
                print(e)


def create_test_data(test_data,testset,cat,IMG_SIZE):
    for category in cat:
        path = os.path.join(testset,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                test_data.append([new_array,class_num])
            except Exception as e:
                pass
                print(e)



def formalize(X,y,data):
    for features , labels in data:
        X.append(features)
        y.append(labels)

def KNN_acc(k_val,Xtr,Xt,ytr,yt):
    knn = KNeighborsClassifier(n_neighbors=k_val) 
    knn.fit(Xtr,ytr)   
    print(knn.score(Xt, yt)," ",k_val)

create_training_data(training_data,dataset,categories,IMG_SIZE)
create_test_data(test_data,testset,categories,IMG_SIZE)
formalize(Xtr,ytr,training_data)
formalize(Xt,yt,test_data)
Xtr=np.array(Xtr).reshape(-1,IMG_SIZE*IMG_SIZE)
Xt=np.array(Xt).reshape(-1,IMG_SIZE*IMG_SIZE)
KNN_acc(k,Xtr,Xt,ytr,yt)

