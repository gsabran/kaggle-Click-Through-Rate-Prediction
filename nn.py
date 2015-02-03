# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:25:31 2015

@author: Pc-stock2
"""
import numpy as np
import copy

class neural_network:
    """ Implementation of the neural network """
    def __init__(self):
        self.theta=[]
    
    def initialize_parameters(self, len_feature, layers=3, nodes_per_layer=2):
        self.theta.append(np.random.randn(nodes_per_layer, len_feature))
        for i in range(1,layers-1): 
            self.theta.append(np.random.randn(nodes_per_layer, nodes_per_layer+1))
        self.theta.append(np.random.randn(1, nodes_per_layer+1))

    def sigmoid(x): 
        return 1/(1+np.exp(-x))
        
    def forward_propagation(self,feature_vector):
        a, z = [], []
        for i in range(len(self.theta)):
            if (i==0): a.append(np.array(feature_vector))
            else: a.append(np.concatenate(([1],sigmoid(z[i-1])),1))
            z.append(np.dot(self.theta[i],a[i]))
        return sigmoid(z[len(self.theta)-1]), a
    
    def predict_proba (self,feature_vector):
        return forward_propagation(self,feature_vector)[0]
             
    def backward_propagation(self,a,actual,pred):
        alpha, cost_wrt_theta = copy.copy(a), copy.copy(self.theta)
        i=len(self.theta)-1
        #compute the partial derivative for each layer
        while (i>=0):
            if (i==(len(self.theta)-1)): 
                alpha[i]=actual-pred
            else:
                if (i==(len(self.theta)-2)): 
                    alpha_temp=(self.theta[i+1][:,1:]*alpha[i+1])
                else : 
                    alpha_temp=np.transpose(np.dot(self.theta[i+1][:,1:],np.transpose(alpha[i+1])))
                alpha[i]=alpha_temp*a[i+1][1:]*(1-a[i+1][1:])
            
            cost_wrt_theta[i]=np.outer(alpha[i],a[i])
            i-=1
        return cost_wrt_theta
            
    
    def update (self,cost_wrt_theta,l_r):
        self.theta = np.add(self.theta,np.multiply(cost_wrt_theta,-l_r))
                         
    def fit(self,x,y,l_r=0.001):
        pred, a = self.forward_propagation(x[0])
        cost_wrt_theta = self.backward_propagation(a,y[0],pred)
        for i in range(1,len(x)):
            pred, a = self.forward_propagation(x[i])
            cost_wrt_theta = np.add(cost_wrt_theta,self.backward_propagation(a,y[i],pred))
        cost_wrt_theta=np.multiply(cost_wrt_theta, 1.0/len(x))
        self.update(cost_wrt_theta,l_r)

    
    