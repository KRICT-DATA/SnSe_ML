import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math

from elements import ELEMENTS
from math import exp


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


def linear(x0,y0,x1,y1,x):
    y= (y1-y0)/(x1-x0)*x + y1-(y1-y0)/(x1-x0)*x1
    return y
    
def root(x0,y0,x1,y1,x):
    c=np.divide(x, np.absolute(x), out=np.zeros_like(x), where=x!=0)
    y= c*(y1-y0)/(np.sqrt(x1)-np.sqrt(x0))*np.sqrt(np.absolute(x)) + y1-(y1-y0)/(np.sqrt(x1)-np.sqrt(x0))*np.sqrt(x1) 
        
    return y


def vbm_ordering(X_train,vbm_e,vbm_e2,vbm_e3,vbm_m,vbm_m2,vbm_m3):
    nTrain=X_train.shape[0]
    
    for i in range(0,nTrain):
        if (X_train[i,vbm_e] > X_train[i,vbm_e2]) and (X_train[i,vbm_e] > X_train[i,vbm_e3]) : 
            X_train[i,vbm_e] = X_train[i,vbm_e]
            X_train[i,vbm_m] = X_train[i,vbm_m]

            if X_train[i,vbm_e2] > X_train[i,vbm_e3] : 
                X_train[i,vbm_e2] = X_train[i,vbm_e2]
                X_train[i,vbm_m2] = X_train[i,vbm_m2]
            else:    
                ffff = X_train[i,vbm_e2]
                ssss = X_train[i,vbm_m2]
                X_train[i,vbm_e3] = X_train[i,vbm_e2]
                X_train[i,vbm_m3] = X_train[i,vbm_m2]
                X_train[i,vbm_e2] = ffff
                X_train[i,vbm_m2] = ssss
                        
        elif (X_train[i,vbm_e] < X_train[i,vbm_e2]) and (X_train[i,vbm_e2] > X_train[i,vbm_e3]) :                         
            ffff = X_train[i,vbm_e]
            ssss = X_train[i,vbm_m]                        
            X_train[i,vbm_e] = X_train[i,vbm_e2]
            X_train[i,vbm_m] = X_train[i,vbm_m2]
                        
            if ffff > X_train[i,vbm_e3] : 
                X_train[i,vbm_e2] = ffff
                X_train[i,vbm_m2] = ssss
            else:    
                X_train[i,vbm_e2] = X_train[i,vbm_e3]
                X_train[i,vbm_m2] = X_train[i,vbm_m3]
                X_train[i,vbm_e3] = ffff
                X_train[i,vbm_m3] = ssss                        
                        
        elif (X_train[i,vbm_e] < X_train[i,vbm_e3]) and (X_train[i,vbm_e2] < X_train[i,vbm_e3]) :                         
            ffff = X_train[i,vbm_e]
            ssss = X_train[i,vbm_m]                        
            X_train[i,vbm_e] = X_train[i,vbm_e3]
            X_train[i,vbm_m] = X_train[i,vbm_m3]
                        
            if ffff > X_train[i,vbm_e2] : 
                X_train[i,vbm_e3] = X_train[i,vbm_e2]
                X_train[i,vbm_m3] = X_train[i,vbm_m2]
                X_train[i,vbm_e2] = ffff
                X_train[i,vbm_m2] = ssss
                        
            else:    
                X_train[i,vbm_e3] = ffff
                X_train[i,vbm_m3] = ssss    
    return X_train


def vbm_ordering_energy(X_train,vbm_e,vbm_e2,vbm_e3):
    nTrain=X_train.shape[0]
    
    for i in range(0,nTrain):
        if (X_train[i,vbm_e] > X_train[i,vbm_e2]) and (X_train[i,vbm_e] > X_train[i,vbm_e3]) : 
            X_train[i,vbm_e] = X_train[i,vbm_e]

            if X_train[i,vbm_e2] > X_train[i,vbm_e3] : 
                X_train[i,vbm_e2] = X_train[i,vbm_e2]
            
            else:    
                ffff = X_train[i,vbm_e2]
                X_train[i,vbm_e3] = X_train[i,vbm_e2]
                X_train[i,vbm_e2] = ffff
            
                        
        elif (X_train[i,vbm_e] < X_train[i,vbm_e2]) and (X_train[i,vbm_e2] > X_train[i,vbm_e3]) :                         
            ffff = X_train[i,vbm_e]
            X_train[i,vbm_e] = X_train[i,vbm_e2]
                        
            if ffff > X_train[i,vbm_e3] : 
                X_train[i,vbm_e2] = ffff
            
            else:    
                X_train[i,vbm_e2] = X_train[i,vbm_e3]
                X_train[i,vbm_e3] = ffff
            
                        
        elif (X_train[i,vbm_e] < X_train[i,vbm_e3]) and (X_train[i,vbm_e2] < X_train[i,vbm_e3]) :                         
            ffff = X_train[i,vbm_e]
            X_train[i,vbm_e] = X_train[i,vbm_e3]

                        
            if ffff > X_train[i,vbm_e2] : 
                X_train[i,vbm_e3] = X_train[i,vbm_e2]
                X_train[i,vbm_e2] = ffff

                        
            else:    
                X_train[i,vbm_e3] = ffff

    return X_train


def cbm_ordering(X_train,cbm_e,cbm_e2,cbm_e3,cbm_m,cbm_m2,cbm_m3):
    nTrain=X_train.shape[0]
    for i in range(0,nTrain):
        if (X_train[i,cbm_e] < X_train[i,cbm_e2]) and (X_train[i,cbm_e] < X_train[i,cbm_e3]) : 


            if X_train[i,cbm_e2] > X_train[i,cbm_e3] : 

                ffff = X_train[i,cbm_e2]
                ssss = X_train[i,cbm_m2]
                X_train[i,cbm_e3] = X_train[i,cbm_e2]
                X_train[i,cbm_m3] = X_train[i,cbm_m2]
                X_train[i,cbm_e2] = ffff
                X_train[i,cbm_m2] = ssss
                        
        elif (X_train[i,cbm_e] > X_train[i,cbm_e2]) and (X_train[i,cbm_e2] < X_train[i,cbm_e3]) :                         
            ffff = X_train[i,cbm_e]
            ssss = X_train[i,cbm_m]                        
            X_train[i,cbm_e] = X_train[i,cbm_e2]
            X_train[i,cbm_m] = X_train[i,cbm_m2]
                        
            if ffff < X_train[i,cbm_e3] : 
                X_train[i,cbm_e2] = ffff
                X_train[i,cbm_m2] = ssss
            else:    
                X_train[i,cbm_e2] = X_train[i,cbm_e3]
                X_train[i,cbm_m2] = X_train[i,cbm_m3]
                X_train[i,cbm_e3] = ffff
                X_train[i,cbm_m3] = ssss                        
                      
        elif (X_train[i,cbm_e] > X_train[i,cbm_e3]) and (X_train[i,cbm_e2] > X_train[i,cbm_e3]) :                         
            ffff = X_train[i,cbm_e]
            ssss = X_train[i,cbm_m]                        
            X_train[i,cbm_e] = X_train[i,cbm_e3]
            X_train[i,cbm_m] = X_train[i,cbm_m3]
                        
            if ffff < X_train[i,cbm_e2] : 
                X_train[i,cbm_e3] = X_train[i,cbm_e2]
                X_train[i,cbm_m3] = X_train[i,cbm_m2]
                X_train[i,cbm_e2] = ffff
                X_train[i,cbm_m2] = ssss
                        
            else:    
                X_train[i,cbm_e3] = ffff
                X_train[i,cbm_m3] = ssss       
                
    return X_train

def cbm_ordering_energy(X_train,cbm_e,cbm_e2,cbm_e3):
    nTrain=X_train.shape[0]
    for i in range(0,nTrain):
        if (X_train[i,cbm_e] < X_train[i,cbm_e2]) and (X_train[i,cbm_e] < X_train[i,cbm_e3]) : 


            if X_train[i,cbm_e2] > X_train[i,cbm_e3] : 

                ffff = X_train[i,cbm_e2]
            
                X_train[i,cbm_e3] = X_train[i,cbm_e2]
            
                X_train[i,cbm_e2] = ffff
            
                        
        elif (X_train[i,cbm_e] > X_train[i,cbm_e2]) and (X_train[i,cbm_e2] < X_train[i,cbm_e3]) :                         
            ffff = X_train[i,cbm_e]
            
            X_train[i,cbm_e] = X_train[i,cbm_e2]
            
                        
            if ffff < X_train[i,cbm_e3] : 
                X_train[i,cbm_e2] = ffff
            
            else:    
                X_train[i,cbm_e2] = X_train[i,cbm_e3]
            
                X_train[i,cbm_e3] = ffff
            
                      
        elif (X_train[i,cbm_e] > X_train[i,cbm_e3]) and (X_train[i,cbm_e2] > X_train[i,cbm_e3]) :                         
            ffff = X_train[i,cbm_e]
            
            X_train[i,cbm_e] = X_train[i,cbm_e3]
            
                        
            if ffff < X_train[i,cbm_e2] : 
                X_train[i,cbm_e3] = X_train[i,cbm_e2]
                
                X_train[i,cbm_e2] = ffff
                
                        
            else:    
                X_train[i,cbm_e3] = ffff
                
    return X_train                