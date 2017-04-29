
# coding: utf-8

# In[ ]:

import pandas as pd
import sklearn
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import itertools
from csv import reader
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


def read_data(file):
    csv_reader = read_csv(file,header=None,delim_whitespace=True)
    data=np.array(csv_reader)
    return data

#this function will check the accuracy 
def accuracy(y_test_actual,y_test_predicted):
    total=len(y_test_predicted);
    #print (total);
    hit=0;
    mis=0;
    for i in range(0,len(y_test_predicted)):
        if(y_test_predicted[i]==y_test_actual[i]):
            hit=hit+1;
        else:
            mis=mis+1;       
    #print('hit = ',hit,' Hit Ratio = ',(hit/float(total)),' mis = ',mis,' Mis Ratio = ',(mis/float(total)));
    return (hit/float(total))
#loading data from csv file

x_train=read_data("data/train/X_train.txt")
y_train=read_data("data/train/y_train.txt")
x_test=read_data("data/test/X_test.txt")
y_test_actual=read_data("data/test/y_test.txt")

#clf
pca = PCA(n_components=2);
plot_x = pca.fit_transform(x_train);
features = 10;
acc=[]
no_features = []

while features <= 561:
    pca = PCA(n_components=features);
    x_train_temp = pca.fit_transform(x_train);
    x_test_temp = pca.transform(x_test);

    #SVC and NuSVC implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification
    clf = svm.SVC()
    clf.fit(x_train_temp, y_train.ravel()) ;
    y_test_predicted=clf.predict(x_test_temp);

    #to check accuracy of  predicted and actual value;
    no_features.append(features);
    acc.append(accuracy(y_test_actual,y_test_predicted))
    if features==510:
        features = 561;
    else:
        features += 50
    #print (features)

#KNeighborsClassifier
pca = PCA(n_components=2);
plot_x = pca.fit_transform(x_train);
features = 10;
acc_2=[]
no_features = []

while features <= 561:
    pca = PCA(n_components=features);
    x_train_temp = pca.fit_transform(x_train);
    x_test_temp = pca.transform(x_test);

    #SVC and NuSVC implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification
    #clf = svm.SVC()
    #clf.fit(x_train_temp, y_train.ravel()) ;
    #y_test_predicted=clf.predict(x_test_temp);
    
    
    neigh=KNeighborsClassifier(n_neighbors=50)
    neigh.fit(x_train_temp,np.ravel(y_train))
    knn_Y=neigh.predict(x_test_temp)

    #to check accuracy of  predicted and actual value;
    no_features.append(features);
    acc_2.append(accuracy(y_test_actual,knn_Y))
    if features==510:
        features = 561;
    else:
        features += 50
    #print (features)

    
#MLPClassifier
pca = PCA(n_components=2);
plot_x = pca.fit_transform(x_train);
features = 10;
acc_4=[]
no_features = []

while features <= 561:
    pca = PCA(n_components=features);
    x_train_temp = pca.fit_transform(x_train);
    x_test_temp = pca.transform(x_test);

    clf=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(80,),random_state=1,activation='relu')
    clf.fit(x_train_temp,np.ravel(y_train))
    y=clf.predict(x_test_temp)

    #to check accuracy of  predicted and actual value;
    no_features.append(features);
    acc_4.append(accuracy(y_test_actual,y))
    if features==510:
        features = 561;
    else:
        features += 50
    #print (features)


plt.plot(no_features,acc, label="MLP");
plt.plot(no_features,acc_2, label="KNN");
#plt.plot(no_features,acc_3, label="RanForest");
plt.plot(no_features,acc_4, label="SVM");
plt.axis([0, 600, 0.7, 1]);
plt.legend();
plt.savefig('features_accuracy.png');
plt.show();