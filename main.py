#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:46:12 2023

@author: shenli
"""

from bayes_opt import BayesianOptimization
from Models.ANN_model import ANN
import numpy as np    
class optimization():
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test       
        self.best_score = 0
    
    
    def ANN_estimator(self,N_layers,N_neuron,batch_size,max_iter,learning_rate,alpha):
############ fix parameters#############
        no_cuda = False
        seed = 1
        log_interval = 2
        test_batch_size = 1000
############ optimized parameters############
        l2_penalty = 0.1**(int(alpha)+1)#0.01,0.001,0.0001,0.00001
        
        lr = 0.1**(int(learning_rate)+1)#0.01,0.001,0.0001,0.00001

        batch_size = 16*(2**int(batch_size))##32,64,128,256 
        
        neuron_size = 8*(2**int(N_neuron))
        
        layer_size = int(N_layers)

        
        epoches = 5*(2**int(max_iter))##10,20,40,80,160,320
        
        # Training with parameters and obtaining score for next step optimization.
        
        model = ANN(self.X_train,self.y_train,self.X_test,self.y_test)
        
        model.feed_para(no_cuda,
                        seed,
                        log_interval,
                        test_batch_size,
                        l2_penalty,
                        lr,
                        batch_size,
                        neuron_size,
                        layer_size,
                        epoches
                        )
        
        score = model.process_training()
        
        if score>self.best_score:
            self.best_score = score
        return score

    
    def fit(self):
        ann_bo = BayesianOptimization(
            self.ANN_estimator,
            {'N_layers':(1,6.99),
             'N_neuron':(3,9.99),
             'batch_size':(1,4.99),
             'learning_rate':(1,4.99),
             'max_iter':(1,6.99),
             'alpha':(1,4.99)
             })
        ann_bo.maximize()    


#####################main#########################
def main():
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    bayes = optimization(X_train,y_train,X_test,y_test)
    bayes.fit()

if __name__ == '__main__':
    main()
