import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import random
import pickle
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
dataset="E:/Data set for pattern/tr"
testset="E:/Data set for pattern/t"
categories =["non-ped_examples","ped_examples"]
training_data=[] 
test_data=[] 
Xtr=[]
ytr=[]
Xt=[]
yt=[]
IMG_SIZE = 50

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

def SVM_Test(Xtr,Xt,ytr,yt):
        clf = SVC(kernel='linear',C=1000.0) 
        # fitting x samples and y classes 
        clf.fit(Xtr,ytr)
        y_pred = clf.predict(Xt)
        print(y_pred)

        print(clf.score(Xt,yt))
        """
        xfit = np.linspace(-1, 3.5)   
        # Y containing two classes 
        Xtr, ytr = make_blobs(n_samples=500, centers=2, 
                        random_state=0, cluster_std=0.40) 
        # plot a line between the different sets of data 
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=50, cmap='spring')
        for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
            yfit = m * xfit + b 
            plt.plot(xfit, yfit, '-k') 
            plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',  
            color='#AAAAAA', alpha=0.4)  
        plt.xlim(-1, 3.5)
        # plotting scatters  
        plt.show()  """

create_training_data(training_data,dataset,categories,IMG_SIZE)
create_test_data(test_data,testset,categories,IMG_SIZE)
formalize(Xtr,ytr,training_data)
formalize(Xt,yt,test_data)
Xtr=np.array(Xtr).reshape(-1,IMG_SIZE*IMG_SIZE)
Xt=np.array(Xt).reshape(-1,IMG_SIZE*IMG_SIZE)
SVM_Test(Xtr,Xt,ytr,yt)
