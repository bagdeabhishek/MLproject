import pandas as pd
import sklearn
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from pandas import read_csv

def read_data(file):
    csv_reader = read_csv(file,header=None,delim_whitespace=True)
    data=np.array(csv_reader)
    return data

#this function will check the accuracy 
def accuracy(y_test_actual,y_test_predicted):
    total=len(y_test_predicted);
    print (total);
    hit=0;
    mis=0;
    for i in range(0,len(y_test_predicted)):
        if(y_test_predicted[i]==y_test_actual[i]):
            hit=hit+1;
        else:
            mis=mis+1;
            
    print('hit = ',hit,' Hit Ratio = ',(hit/float(total)),' mis = ',mis,' Mis Ratio = ',(mis/float(total)));
    
#loading data from csv file
#x_train=np.array(x_train);

#y_train=pd.read_csv('y_train.csv');
#y_train=np.array(y_train);

#x_test=pd.read_csv('X_test.csv');
#x_test=np.array(x_test);

#y_test_actual = pd.read_csv('y_test.csv');
#y_test_actual=np.array(y_test_actual);

x_train=read_data("data/train/X_train.txt")
y_train=read_data("data/train/y_train.txt")
x_test=read_data("data/test/X_test.txt")
y_test_actual=read_data("data/test/y_test.txt")


#PCA for the dataset
pca = PCA(n_components=3);
plot_x = pca.fit_transform(x_train);
pca = PCA(n_components=400);
x_train = pca.fit_transform(x_train);
x_test = pca.transform(x_test);

clf = svm.SVC()
clf.fit(x_train, y_train.ravel()) ;
y_test_predicted=clf.predict(x_test);

a_x1=[];
a_x2=[];
b_x1=[];
b_x2=[];
c_x1=[];
c_x2=[];
d_x1=[];
d_x2=[];
e_x1=[];
e_x2=[];
f_x1=[];
f_x2=[];

print(plot_x.shape);
for i in range (plot_x.shape[0]):
    if y_train[i] ==1:
        a_x1.append(plot_x[i][0]); 
        a_x2.append(plot_x[i][1])
    elif y_train[i] ==2:
        b_x1.append(plot_x[i][0])
        b_x2.append(plot_x[i][1])
    elif y_train[i]==3:
        c_x1.append(plot_x[i][0])
        c_x2.append(plot_x[i][1])
    elif y_train[i]==4:
        d_x1.append(plot_x[i][0])
        d_x2.append(plot_x[i][1])
    elif y_train[i]==5:
        e_x1.append(plot_x[i][0])
        e_x2.append(plot_x[i][1])
    elif y_train[i]==6:
        f_x1.append(plot_x[i][0])
        f_x2.append(plot_x[i][1])

plt.plot(a_x1,a_x2,'bo');
plt.plot(b_x1,b_x2,'ro');
plt.plot(c_x1,c_x2,'yo');
plt.plot(d_x1,d_x2,'ko');
plt.plot(e_x1,e_x2,'co');
plt.plot(f_x1,f_x2,'mo');
plt.savefig('PCA.png');
plt.show();
