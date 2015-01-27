# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:05:31 2015

@author: Pc-stock2
"""



import os
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier



#find the mean, minimum and maximum values for feature scaling
def feature_scaling():
    n_rows=0
    for df in pd.read_csv('train.csv',chunksize=100000):
        if (n_rows==0):
            global total, minimum, maximum
            total=sum(df.values)
            minimum=df.values.min(axis=0)
            maximum=df.values.max(axis=0)
        else:
            total += sum(df.values)
            minimum = np.array((minimum,df.values.min(axis=0))).min(axis=0)
            maximum = np.array((maximum,df.values.max(axis=0))).max(axis=0)
            n_rows += len(df)

    x_mean=total[1:]/n_rows
    x_range=maximum[1:]-minimum[1:]
    #temporary - forgot to add the intercept in the processing
    #that's because using ski-learn, you don't need to manually add it
    x_mean = np.concatenate((x_mean,[0]),1)
    x_range = np.concatenate((x_range,[1]),1)
    ####
    return x_mean, x_range


def proba_y(x,theta):
    return 1/(1+np.exp(-np.dot(x,theta)))

def update_y(x,y,p_y,alpha,n):
    return alpha*np.sum((p_y-y)*x,1)/n

def logloss(y,p_y,n):
    return -(np.dot(y,np.log(p_y))+ np.dot((1-y),np.log(1-p_y)))/n

##==========================================
#using skit-learn
#set lr = SGDClassifier(loss="log")
#define classes=[0,1]
#predict the proba using lr.predict_proba(x)[:,1]
#update the parameters using lr.partial_fit(x,y,classes=classes)
#lr.coef_, lr.intercept_ to get the parameters and intercept

#however using skit-learn I am not able to find convergence
#so decided to define everything myself
##=========================================  


def logistic_regression(filename,rounds,x_mean,x_range,chunk_size,alpha,start_theta):
    t0 = time.time()
    theta = start_theta
    counter, cost= 1, 0
    mean_cost = []
    for i in range(rounds):
        for df in pd.read_csv(filename,chunksize=chunk_size):
            #temporary:
            df['intercept']=1
            ##########
            x = (df.ix[:,1:].values-x_mean)/x_range
            y = df.ix[:,0].values
            #record the performance for the last two rounds to see if it converges
            if (i in [rounds-2,rounds-1]):
                p_y = proba_y(x,theta)
                cost += logloss(y,p_y,len(df))
                # every 5000 chunks (500,000 rows) record the mean performance
                if (counter==5000): 
                    mean_cost += [cost/(counter*chunk_size)]
                    counter, cost= 0, 0
                    #print mean_cost[len(mean_cost)-1], theta
            theta -= update_y(x.T,y,p_y,alpha,len(df))
            counter += 1
        print mean_cost
    t1=time.time()
    print (t1-t0)
    return mean_cost, theta

#############################################
#############################################

start_theta=np.array([random.random() for i in range(24)])
alpha=0.0001
chunk_size=100

x_mean, x_range = feature_scaling()
a, b = logistic_regression('train.csv',2,x_mean,x_range,chunk_size,alpha,start_theta)

plt.plot(a) 
# --> going through the file twice, we find that the mean cost decreases



#####check how we do on the validation set using the estimated parameters

#make a few transformations to the validation set
valid=pd.read_csv('valid.csv')
valid_nrows=len(valid)
valid_total=sum(valid.ix[:,1:].values)
valid_minimum=valid.ix[:,1:].values.min(axis=0)
valid_maximum=valid.ix[:,1:].values.max(axis=0)

valid_mean=valid_total/valid_nrows
valid_range=valid_maximum-valid_minimum

valid_mean = np.concatenate((valid_mean,[0]),1)
valid_range = np.concatenate((valid_range,[1]),1)

valid['intercept']=1 #temporary
x_valid=(valid.ix[:,1:].values-valid_mean)/valid_range 
y_valid=valid.ix[:,0].values

valid_proba_y= proba_y(x_valid,b)
logloss(y_valid,valid_proba_y,valid_nrows)

