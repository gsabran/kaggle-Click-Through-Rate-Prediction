# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:25:31 2015

@author: Pc-stock2
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

#IMPORTANT:
#always use matrices - shapes of the type (NL,) are confusing
#be careful with the copy.deepcopy()

def sigmoid(x): 
    return 1.0/(1+np.exp(-x))

class neural_network:
    """ Implementation of the neural network """
    def __init__(self):
        self.theta=[]
        self.initialized = False
        #for checking purposes:
        self.theta_epsilon=[]
        self.grad_checking=[]
        self.cost_checking=[]
    
    def initialize_parameters(self, len_feature, layers=3, nodes_per_layer=2):
        self.theta.append(np.random.randn(nodes_per_layer, len_feature))
        if (layers>0):
            for i in range(1,layers-1): 
                self.theta.append(np.random.randn(nodes_per_layer, nodes_per_layer+1))
            self.theta.append(np.random.randn(1, nodes_per_layer+1))
        self.initialized = True
      
    def forward_propagation(self,feature_vector,actual=0,verify_gradient=False):
        a, z = [], []
        cost = 0
        t = copy.deepcopy(self.theta)
        if (verify_gradient is True): t = copy.deepcopy(self.theta_epsilon)
        for i in range(len(t)):
            if (i==0): a.append(np.transpose(np.matrix(feature_vector)))
            else: a.append(np.concatenate(([[1]], sigmoid(z[i-1])),axis=0))
            z.append(np.dot(t[i],a[i]))
        pred = sigmoid(z[len(t)-1])
        cost = -(actual*np.log(pred) + (1-actual)*np.log(1-pred))
        return pred, cost, a       
    
    def predict_proba (self,feature_vector):
        return self.forward_propagation(feature_vector)[0]
             
    def backward_propagation(self,a,actual,pred):
        #compute the partial derivative for each layer
        alpha, cost_wrt_theta = copy.deepcopy(a), copy.deepcopy(self.theta)
        i=len(self.theta)-1
        while (i>=0):
            if (i==(len(self.theta)-1)): 
                alpha[i]=pred-actual
            else :
                alpha_temp=np.transpose(self.theta[i+1][:,1:])*alpha[i+1]
                alpha[i]=np.multiply(alpha_temp,np.multiply(a[i+1][1:,],(1-a[i+1][1:,]))) 
                
            cost_wrt_theta[i]=np.outer(alpha[i],a[i])
            i-=1
        return cost_wrt_theta
    
       
    def check_backward_propagation(self,a, actual, pred, feature_vector, e=0.00001):
        #compute the gradient
        grad_difference=self.backward_propagation(a,actual,pred)
        #estimation of gradient numerically
        for i in range(np.shape(self.theta)[0]):
            for j in range(np.shape(self.theta[i])[0]):
                for k in range(np.shape(self.theta[i][j])[0]):
                    loss = []
                    for sign in [1, -1]:
                        self.theta_epsilon = copy.deepcopy(self.theta)
                        self.theta_epsilon[i][j,k] += sign*e
                        loss.append(self.forward_propagation(feature_vector,actual,True)[1])
                    print grad_difference[i][j,k], (loss[0]-loss[1])/(2*e)
                    #grad_difference[i][j,k] -= (loss[0]-loss[1])/(2*e)
                    
                           
    def update (self,cost_wrt_theta,l_r):
        self.theta = np.add(self.theta, np.multiply(cost_wrt_theta,-l_r))
                         
    def fit(self,x,y,l_r=0.001,compute_cost=True,verify_gradient=False):
        pred, cost, a = self.forward_propagation(x[0],y[0])
        cost_wrt_theta = self.backward_propagation(a,y[0],pred)
        for i in range(1,len(x)):
            pred, new_cost, a = self.forward_propagation(x[i],y[i])
            if (compute_cost is True): cost += new_cost
            if (verify_gradient is True): self.check_backward_propagation(a, y[i], pred, x[i])
            cost_wrt_theta = np.add(cost_wrt_theta,self.backward_propagation(a,y[i],pred))
        self.cost_checking.append(cost/len(x))
        cost_wrt_theta=np.multiply(cost_wrt_theta,(1.0/len(x)))
        self.update(cost_wrt_theta,l_r)
    
    def print_cost_checking(self):
        plt.figure()
        plt.plot(self.cost_checking)


    
    