#!/usr/bin/python

import csv
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

count = 0
sample=[]
X_pca=[]
predicted=[]
total_samples = 0
TN = 0.0
TP = 0.0
FN = 0.0
FP = 0.0
def confusion_matrixt(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def print_metrics(confusion_matrix):
    confusion_matrix=np.array(confusion_matrix)
    confusion_matrix=confusion_matrix.astype(np.float)
    print "confusion matrix:"
    print confusion_matrix
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    Accuracy = ((TN+TP)/(TN+TP+FN+FP))*100.0
    TPR = ((TP)/(TP+FN))*100.0
    PPV = ((TP)/(TP+FP))*100.0
    TNR = ((TN)/(TN+FP))*100.0
    print "Accuracy:%f" % (Accuracy)
    print "TPR:%f" % (TPR)
    print "PPV:%f" % (PPV)
    print "TNR:%f" % (TNR)

 
with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

X=np.array(sample)
Y=X.astype(np.float)

X_tr = Y[0:226]
X_va = Y[227:453]
X_te = Y[454:680]


X_label=X_tr[:,X_tr.shape[1]-1]
Xtr_label=X_label.reshape(X_tr.shape[0],1)
Xtr_data=X_tr[:,:X_tr.shape[1] - 1]

def kNN(K):
    correctly_classified = 0
    total_samples = X_te.shape[0]
    class0 = 0
    class1 = 0
    predicted=[]

#Uncomment below line to test on test dataset and comment the "for loop below to it"
#    for i in xrange(X_te.shape[0]):
    for i in xrange(X_va.shape[0]):
#Uncomment below line to test on test dataset and comment the "below to it"
#       X_tmp       = X_te[i]
        X_tmp       = X_va[i]
        Xte_label   = X_tmp[-1]
        X_tmp       = X_tmp[:-1]
        Xte_data    = np.tile(X_tmp, (X_tr.shape[0],1)) 
        y           = Xtr_data - Xte_data
        y           = np.square(y)
        y           = np.sum(y,axis=1)
        y           = np.sqrt(y)
        y1          = y.reshape(Xtr_data.shape[0],1)
        y2          = np.c_[y1, Xtr_label]
        y2          = y2[y2[:,0].argsort()]
        K_samples   = y2[:K]
        
        for j in xrange(K):
            temp=K_samples[j]
            if (temp[-1] == 2.0):
                class0+=1
            else:
                class1+=1

        if (class0 > class1):
            predicted.append(2.0)
            if Xte_label == 2.0:
                correctly_classified+=1
        else:
            predicted.append(4.0)
            if Xte_label == 4.0:
                correctly_classified+=1

        class0 = 0
        class1 = 0
    return correctly_classified, predicted, total_samples


#######################################################
# You need to change X_va to X_te when testing set is used
# to generate confusion matrix
#######################################################
actual=[row[-1] for row in X_va]
#print actual 
#print predicted 


#######################################################
# Uncomment this section to try out for different K values
#######################################################
"""
count = 3
while(count < 34):
    correct,predicted_temp,total_sample_count=kNN(count)
    confusion_matrix=confusion_matrixt(actual, predicted_temp)
    print_metrics(confusion_matrix)
    correctly_classified=float(correct)
    Accuracy = (correctly_classified/total_sample_count)*100
    print "%d %f" % (count, Accuracy)
    count+=2
"""
#######################################################
# To test please supple the correct kNN Value for my case
# when K = 11 my algorithm performs best both on test and validation
# dataset
#######################################################
K=11
if len(sys.argv)==3:
    K=int(sys.argv[2])
correct,predicted_temp,total_sample_count=kNN(K)
confusion_matrix=confusion_matrixt(actual, predicted_temp)
print_metrics(confusion_matrix)
correctly_classified=float(correct)
Accuracy = (correctly_classified/total_sample_count)*100
print "%d %f" % (K, Accuracy)
            
             
        
        

    








