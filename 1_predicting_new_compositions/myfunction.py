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

        
def tename(GBR,x_trial,indexing):
    
    
    ylegend = str(GBR)
    ylegend += ',Sn'
    if x_trial[0][0] != 1:
        ylegend += str(x_trial[0][0])
        
    if ELEMENTS[x_trial[0][2]] != 0:
        ylegend += str(ELEMENTS[x_trial[0][2]].symbol)
        ylegend += str(x_trial[0][3])
        if x_trial[0][4] != 0:
            ylegend += str(ELEMENTS[x_trial[0][4]].symbol)
            ylegend += str(x_trial[0][5])
            if x_trial[0][6] != 0:
                ylegend += str(ELEMENTS[x_trial[0][6]].symbol)
                ylegend += str(x_trial[0][7])
    ylegend += 'Se'
    if x_trial[0][1] != 1:
        ylegend += str(x_trial[0][1])
    
    if not (indexing is ''):
        ylegend += ','
        ylegend += str(indexing)

    return ylegend
    
def tename1(GBR,x_trial,indexing):
    
    
    ylegend = str(GBR)
    ylegend += ',Sn'
    if x_trial[0] != 1:
        ylegend += str(x_trial[0][0])
        
    if ELEMENTS[x_trial[2]] != 0:
        ylegend += str(ELEMENTS[x_trial[2]].symbol)
        ylegend += str(x_trial[3])
        if x_trial[4] != 0:
            ylegend += str(ELEMENTS[x_trial[4]].symbol)
            ylegend += str(x_trial[5])
            if x_trial[6] != 0:
                ylegend += str(ELEMENTS[x_trial[6]].symbol)
                ylegend += str(x_trial[7])
    ylegend += 'Se'
    if x_trial[0][1] != 1:
        ylegend += str(x_trial[1])
    
    if not (indexing is ''):
        ylegend += ','
        ylegend += str(indexing)

    return ylegend
        
def readexp(thermo,snratio,seratio,dopant1,dopant1ratio,dopant2,dopant2ratio,dopant3,dopant3ratio,direction):
    xxx=[]
    yyy=[]
    for i in range(thermo.shape[0]):            
    # thermo.iloc[i,1] == 'Sn ratio', iloc[i,2] == 'Se ratio', iloc[i,3] == 'dopant1 name', iloc[i,5] == 'dopant1 ratio',
    # iloc[i,6] =='dopant2 name', iloc[i,8]=='dopant2 ratio', iloc[i,9] == 'dopant3 name', iloc[i,11] == 'dopant3 ratio' 
    # iloc[i,12] == direction
    
        if thermo.iloc[i,1] == snratio and thermo.iloc[i,2] == seratio and (thermo.iloc[i,3] == dopant1 or thermo.iloc[i,4] == dopant1) \
           and thermo.iloc[i,5] == dopant1ratio and (thermo.iloc[i,6] == dopant2 or thermo.iloc[i,7] == dopant2) and thermo.iloc[i,8] == dopant2ratio \
           and (thermo.iloc[i,9] == dopant3 or thermo.iloc[i,10] == dopant3) and thermo.iloc[i,11] == dopant3ratio and thermo.iloc[i,12] == direction:
        
            xxx.append(thermo.iloc[i,13])    # temperature
            yyy.append(thermo.iloc[i,14])   # electronic conductivity
    
    ylegend = tename('exp',[[snratio,seratio,dopant1,dopant1ratio,dopant2,dopant2ratio,dopant3,dopant3ratio,direction],[]],'')
  
    plt.plot(xxx,yyy,label=ylegend,marker='o') 

def linear(x0,y0,x1,y1,x):
    y= (y1-y0)/(x1-x0)*x + y1-(y1-y0)/(x1-x0)*x1
    return y
    
def root(x0,y0,x1,y1,x):
    c=np.divide(x, np.absolute(x), out=np.zeros_like(x), where=x!=0)
    y= c*(y1-y0)/(np.sqrt(x1)-np.sqrt(x0))*np.sqrt(np.absolute(x)) + y1-(y1-y0)/(np.sqrt(x1)-np.sqrt(x0))*np.sqrt(x1) 
        
    return y


def ylabelname(measure):
    if measure == 'thermal_conductivity':
        plt.ylabel('Thermal conductivity (W/mK)')
    
    elif measure == 'electrical_conductivity':
        plt.ylabel('Electrical conductivity (S/cm)')
    
    elif measure == 'seebeck_coeff':
        plt.ylabel('Seebeck coefficient (\u03BCV/K)')
    
    return

       
        
def writeid(xxx,Xnum):
    l=[]
    l.append(xxx)
    l.append('Sn')
    l.append(str(Xnum[0]))
    l.append('Se')
    l.append(str(Xnum[1]))
    if Xnum[2]  !=0:
        l.append(ELEMENTS[Xnum[2]].symbol)
        l.append(str(Xnum[3]))
    if Xnum[4]  !=0:
        l.append(ELEMENTS[Xnum[4]].symbol)
        l.append(str(Xnum[5]))
    if Xnum[6]  !=0:
        l.append(ELEMENTS[Xnum[6]].symbol)
        l.append(str(Xnum[7]))
    id=''.join(l)   
    return id



def figsetting(measure, fig, ax):
    
    if measure == 'thermal_conductivity':
        
        ax.set_xlim((0,1.5))
        ax.set_ylim((0,1.5))
        ax.set_xlabel('measured thermal conductivity')
        ax.set_ylabel('predicted thermal conductivity')
        
    elif measure == 'electrical_conductivity':
        
        ax.set_xlim((0,18000))
        ax.set_ylim((0,18000))
        ax.set_xlabel('measured electrical conductivity')
        ax.set_ylabel('predicted electrical conductivity')
        
    elif measure == 'seebeck_coeff':
        
        ax.set_xlim((-600,600))
        ax.set_ylim((-600,600))
        ax.set_xlabel('measured seebeck coefficient')
        ax.set_ylabel('predicted seebeck coefficient')
        
    
    x = np.linspace(-20000, 20000, 10)
    ax.plot(x,x,color='black')
    ax.set_aspect('equal')
    return 


def renorm(Y_train_out, Y_train_out_org, nTrain, Y_diff, Y_mean):

    for j in range(nTrain):
        Y_train_out_org[j] = Y_train_out[j] * Y_diff
        Y_train_out_org[j] += Y_mean     
        
    return 


# count number of sample IDs
def countID(xxx, nTrain):
    nsample=2
    count=[]
    elements=0
    
    for i in range(nTrain-1):
        if xxx[i] != xxx[i+1] :
            nsample += 1
            elements += 1
            count.append(elements)
            elements=0
        else:
            elements += 1
        
        
    count.append(elements+1)        

    idnumber=np.zeros((nsample))
    idnumber=np.array(count)
    
    return idnumber

def selectID_train(train, idnumber):
    
    kk=0
    train_temp=[]
    for j in range(len(train)):
        start=0
        for ll in range(train[j]):
            start += idnumber[ll]
        
        
        for mm in range(idnumber[train[j]]):
            train_temp.append(start + mm)
            kk+=1
            

    train_real=np.zeros((len(train_temp)))
    train_real=np.array(train_temp)
    
    return train_real


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