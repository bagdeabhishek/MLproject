
# coding: utf-8

# In[8]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from csv import reader
from pandas import read_csv
import sklearn as skl
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# global class_names = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING", "LAYING"]
def read_data(file):
    csv_reader = read_csv(file,header=None,delim_whitespace=True)
    data=np.array(csv_reader)
    return data

def svm_class(train_x,train_y,test_x,test_y):
    for kernel in {'linear','poly','rbf'}:
        clf=svm.SVC(verbose=True,kernel=kernel)
    # print(train_x.shape,train_y.shape)
        clf.fit(train_x,np.ravel(train_y))
        Y=clf.predict(test_x)
        print("SVM ",kernel,":",accuracy_score(Y,test_y))
        print(classification_report(Y,test_y))
        print(confusion_matrix(test_y,Y))
    
def KNN(train_x,train_y,test_x,test_y):
    neigh=KNeighborsClassifier(n_neighbors=50)
    neigh.fit(train_x,np.ravel(train_y))
    knn_Y=neigh.predict(test_x)
#     print("KNN: ",mean_squared_error(knn_Y,test_y))
    print("KNN: ",accuracy_score(knn_Y,test_y))
    print(classification_report(knn_Y,test_y))
    print(confusion_matrix(test_y,knn_Y))

def rdforest(train_x,train_y,test_x,test_y):
    rdforest= RandomForestClassifier(n_estimators=100,verbose=True)
    rdforest.fit(train_x,np.ravel(train_y))
    rdforest_y=rdforest.predict(test_x)
#     print("Random Forest: ",mean_squared_error(rdforest_y,test_y))
    print("Random Forest: ",accuracy_score(rdforest_y,test_y))
    print(classification_report(rdforest_y,test_y))
    print(confusion_matrix(test_y,rdforest_y))

def bagging_class(train_x,train_y,test_x,test_y):
    bag= BaggingClassifier(svm.SVC(kernel='linear'))
    bag.fit(train_x,np.ravel(train_y))
    bag_y=bag.predict(test_x)
    print("Bagging Classifier: ",accuracy_score(bag_y,test_y))
    print(classification_report(bag_y,test_y))
    print(confusion_matrix(test_y,bag_y))

def gradboost_class(train_x,train_y,test_x,test_y):
    classif = GradientBoostingClassifier(verbose=True)
    classif.fit(train_x,np.ravel(train_y))
    y=classif.predict(test_x)
#     print(classif.estimators_)
    print("GradBoost Classifier: ",accuracy_score(y,test_y))
    print(classification_report(y,test_y))
    print(confusion_matrix(test_y,y))

    
def perceptron_nn(train_x,train_y,test_x,test_y):
    clf=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(80,),random_state=1,activation='relu')
    clf.fit(train_x,np.ravel(train_y))
    y=clf.predict(test_x)
#     print("perceptron_nn:",mean_squared_error(y,test_y))
    print("perceptron_nn:",accuracy_score(y,test_y))
    print(classification_report(y,test_y))
    
def feature_importance(train_x,train_y,test_x,test_y):
    forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

    forest.fit(train_x,train_y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
#     print("Feature ranking:")
    weight=np.ones((train_x.shape[0],1))
    train_x_red=np.ones(train_x.shape[1])
    print(train_x_red.shape)
#     for f in range(train_x.shape[1]):
#         print("%d. feature          %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    train_x_red=train_x[:,indices[range(1,500)]]
    test_x_red=test_x[:,indices[range(1,500)]]
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(train_x.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(train_x.shape[1]), indices)
    plt.xlim([-1, train_x.shape[1]])
#     plt.show()
    return train_x_red,test_x_red

def PCAnalysis(train_x,train_y,test_x,test_y):
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange','black','red','yellow']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')

    plt.show()

def GausDA(train_x,train_y,test_x,test_y):
    clf=QuadraticDiscriminantAnalysis()
    clf.fit(train_x,np.ravel(train_y))
    y=clf.predict(test_x)
#     print("perceptron_nn:",mean_squared_error(y,test_y))
    print("QuadraticDiscriminant Analysis:",accuracy_score(y,test_y))
    print(classification_report(y,test_y))
    print(confusion_matrix(test_y,y))
    
def LinDA(train_x,train_y,test_x,test_y):
    clf=LinearDiscriminantAnalysis()
    clf.fit(train_x,np.ravel(train_y))
    y=clf.predict(test_x)
#     print("perceptron_nn:",mean_squared_error(y,test_y))
    print("Linear Discriminant analysis:",accuracy_score(y,test_y))
    print(classification_report(y,test_y))
    print(confusion_matrix(test_y,y))
    
    
def PrinCompAna(train_x,train_y,test_x,test_y):
    clf=PCA(n_components=500)
    clf.fit(train_x)
    return(clf.transform(train_x),clf.transform(test_x))


    
    

    
train_x=read_data("data/train/X_train.txt")
train_y=read_data("data/train/y_train.txt")
test_x=read_data("data/test/X_test.txt")
test_y=read_data("data/test/y_test.txt")
scaler=StandardScaler()
scaler.fit(train_x)
train_x=scaler.transform(train_x)
test_x=scaler.transform(test_x)

train_x,test_x=feature_importance(train_x,train_y,test_x,test_y)
# train_x,test_x=PrinCompAna(train_x,train_y,test_x,test_y)
gradboost_class(train_x,train_y,test_x,test_y)

bagging_class(train_x,train_y,test_x,test_y)
KNN(train_x,train_y,test_x,test_y)
rdforest(train_x,train_y,test_x,test_y)
svm_class(train_x,train_y,test_x,test_y)
perceptron_nn(train_x,train_y,test_x,test_y)
# # PCAnalysis(train_x,train_y,test_x,test_y)
# # print(train_x.shape,train_y.shape)
GausDA(train_x,train_y,test_x,test_y)
LinDA(train_x,train_y,test_x,test_y)

# for 95 accuracy 80 neurons
    


# In[ ]:



