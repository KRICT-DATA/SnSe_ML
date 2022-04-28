#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### ten-foled modeling : Gradient Boosting Tree Regression
from sklearn.model_selection import KFold
from sklearn import linear_model
from myfunction import countID
from myfunction import selectID_train
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
from myfunction import figsetting
from sklearn.metrics import mean_absolute_error


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from elements import ELEMENTS

import copy
import math
import sys
import scipy.signal

from math import exp
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from elements import ELEMENTS
from myfunction import my_range
from myfunction import tename
from myfunction import readexp
from myfunction import linear
from myfunction import ylabelname
from myfunction import writeid
from myfunction import renorm

from feature import feature_engineer
from feature import get_calvalues
from feature import construct_features
from feature import onehotencoder

import pickle
from joblib import dump, load


def gbrt(X_train_in,Y_train_in, X_test_in, Y_test_in,thermo,measure,kkk):


    nTrain=X_train_in.shape[0]
    nTest =Y_test_in.shape[0]
    nFeat =X_train_in.shape[1]
    
    X=thermo.drop(['ID','K','dopant1','dopant2','dopant3'],axis=1)
    
        
    nModel=10 
    seed=0

    xxx=thermo.ID

    # model construct
    #GBRT
    clf = GradientBoostingRegressor(n_estimators=3000, \
                                    max_leaf_nodes= 1000, \
                                    max_depth= None, \
                                    random_state= 2, \
                                    min_samples_split=5, \
                                    learning_rate=0.1,   \
                                    subsample=0.8,      \
                                    max_features=  2)


    Y_train_out = np.zeros((nTrain))
    Y_test_out  = np.zeros((nTest))


    #count IDs
    idnumber=countID(xxx,nTrain)

    kf = KFold(n_splits=nModel, random_state=None)


    # split IDs for train and test sets for cross-validation 
    for i, (train, test) in enumerate(kf.split(idnumber,idnumber)):
    
    
        # select train set for cross-validation
        train_real = selectID_train(train,idnumber)

        # insert the train set for chosen iDs
        X_train_model = np.zeros((train_real.shape[0],nFeat))
        Y_train_model = np.zeros((train_real.shape[0]))
    
        X_train_model, Y_train_model = shuffle(X_train_in[train_real], Y_train_in[train_real])
    
    
        #fit models and obtain predicted values
        
        clf.fit(X_train_model, Y_train_model)
        Y_train_out    +=  clf.predict(X_train_in)
        Y_test_out     +=  clf.predict(X_test_in)   

    
    Y_train_out     /=  nModel
    Y_test_out      /=  nModel
    
    filename = str(measure) + '_'+str(kkk)+".joblib"
    dump(clf, filename)

#    rmse_train = np.sqrt(mean_squared_error(Y_train_in,Y_train_out))
#    mae_train = mean_absolute_error(Y_train_in,Y_train_out)
#    r2score_train = r2_score(Y_train_out, Y_train_in)

 
#    rmse_test = np.sqrt(mean_squared_error(Y_test_in, Y_test_out))
#    mae_test = mean_absolute_error(Y_test_in, Y_test_out)
#    r2score_test = r2_score(Y_test_out, Y_test_in)



    if kkk == 4:
        
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)

        column=[]
        column=['direction', 'category', 'Temp', 'weighted_dopant_ratio','weighted_ion_energies','sum_ion_energies','weighted_covalent_radius']
                #'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',\
                #'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',\
                #'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','55','56','57','58','59','60',\
                #'61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80',\
                #'81','Pb','Bi']
            #'ion', 'therm','specific_heat','atomic_radius',\#'weighted_ionenergies','weighted_mass',\'weighted_heat_of_formation']
            
        
        cutoff=20
        if nFeat > cutoff:
            sorted_idx2=np.zeros((cutoff,), dtype=int)
            for ii in range(cutoff):
                sorted_idx2[cutoff-ii-1] = sorted_idx[sorted_idx.shape[0]-ii-1]
            pos = np.arange(sorted_idx2.shape[0]) + .5

            plt.subplot(1, 2, 2)
            plt.barh(pos, feature_importance[sorted_idx2], align='center')

            temp=[]
            for j in range(cutoff):
    
                temp.append(column[sorted_idx2[j]])
    
        else :
            pos = np.arange(sorted_idx.shape[0]) + .5
            plt.subplot(1, 2, 2)
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            temp=[]
            for j in range(nFeat):
                temp.append(column[sorted_idx[j]])



        plt.yticks(pos,(temp))
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()  
    
        
    return Y_train_out, Y_test_out   #, rmse_train, mae_train, r2score_train, rmse_test,mae_test, r2score_test
      



   # figure for test set with IDs

def drawfigure(Y_train_in, Y_test_in, Y_train_out_org, Y_fitted_org, thermo, testset, measure,nFeat):
    
    
    X=thermo.drop(['ID','K','dopant1','dopant2','dopant3'],axis=1)
    XX = testset.drop(['ID','K','dopant1','dopant2','dopant3'],axis=1)
    nTrain = Y_train_in.shape[0]
    nTest = Y_test_in.shape[0]
    
    
    fig = plt.figure(figsize=(20,20))
    ax0=fig.add_subplot(4,2,1)
    ax1=fig.add_subplot(4,2,2)
    ax2=fig.add_subplot(4,2,3)
    ax3=fig.add_subplot(4,2,4)
    ax4=fig.add_subplot(4,2,5)
    ax5=fig.add_subplot(4,2,6)
    ax6=fig.add_subplot(4,2,7)
    ax7=fig.add_subplot(4,2,8)
    #ax8=fig.add_subplot(3,3,9)

    figsetting(measure, fig, ax0)
    figsetting(measure, fig, ax1)
    figsetting(measure, fig, ax2)
    figsetting(measure, fig, ax3)
    figsetting(measure, fig, ax4)
    figsetting(measure, fig, ax5)
    figsetting(measure, fig, ax6)
    figsetting(measure, fig, ax7)
    #figsetting(measure, fig, ax8)

    #label the ID
    #xxx=testset.ID
    xxx=thermo.ID
    yyy=[]
    zzz=[]
    sss=thermo.category
    ddd = thermo.direction

    dopant_test = np.zeros((nTrain,nFeat))
    dopant_test = X.values

    n=0
    mm=0
#overlap = [name for name in mcd.CSS4_COLORS
#           if "xkcd:" + name in mcd.XKCD_COLORS]
    color_array = ['red','orange','yellow','green','aqua','blue','navy','purple','pink','white','tan','silver','violet','lightgreen',               'lime','plum','azure']

    for i in range(nTrain-1):
        if xxx[i] != xxx[i+1] or ddd[i]!=ddd[i+1]:    
        
            yyy.append(Y_train_in[i])
            zzz.append(Y_train_out_org[i])
            if sss[i] == 0 :
                ax0.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            elif sss[i] == 1 :
                
            #if n < 25:
                ax1.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            #else:
            #    ax2.scatter(yyy,zzz,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            #n+=1
            elif sss[i] == 2:
            #if mm <25:
                ax2.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            #else:
            #    ax4.scatter(yyy,zzz,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
                mm+=1
            elif sss[i] == 3 :
                ax3.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            elif sss[i] == 4 :
                ax4.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            elif sss[i] == 5 :
                ax5.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            else:
                ax0.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            yyy=[]
            zzz=[]        
        else:
                
        #plt.scatter(yyy,zzz,alpha=1.0, edgecolors='black',label=writeid(xxx[i],dopant_test[i]))

        
            yyy.append(Y_train_in[i])
            zzz.append(Y_train_out_org[i])


    yyy.append(Y_train_in[nTest-1])
    zzz.append(Y_train_out_org[nTest-1])        
    if sss[i] == 0 :
        ax0.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    elif sss[i] == 1 :
        ax1.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    elif sss[i] == 2:
        ax2.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    elif sss[i] == 3 :
        ax3.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    elif sss[i] == 4 :
        ax4.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    elif sss[i] == 5 :
        ax5.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
    else:
        ax0.scatter(zzz,yyy,alpha=1.0,edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
        
    ax0.legend(loc='center left', bbox_to_anchor=(-1.5, 0.8))         
    ax2.legend(loc='center left', bbox_to_anchor=(-1.5, 0.5))         
    ax4.legend(loc='lower left', bbox_to_anchor=(-1.5, -0.2))
    ax6.legend(loc='lower left', bbox_to_anchor=(-1.5, -1.2))

    ax1.legend(loc='center left', bbox_to_anchor=(1, 1.3)) 
    ax3.legend(loc='lower left', bbox_to_anchor=(1, 0.0))
    ax5.legend(loc='lower left', bbox_to_anchor=(1, -3.0))
    ax7.legend(loc='lower left', bbox_to_anchor=(1, 0.0))
    plt.show()

    rmse = np.sqrt(mean_squared_error(Y_train_in, Y_train_out_org))
    print("RMSE: %.4f" % rmse)
    mae = mean_absolute_error(Y_train_in, Y_train_out_org)
    print("MAE : %.4f" % mae)
#print('number of test samples:',mm*10+n) 


    for i in range(nTest):
        if Y_train_in[i]*Y_train_out_org[i] < 0:
        
            print('+-', writeid(xxx[i],dopant_test[i]))
        elif np.abs(Y_train_in[i] - Y_train_out_org[i]) > rmse*2.0:
            print('>2', writeid(xxx[i],dopant_test[i]))
 

    

# figure for test set with IDs

    fig, ax = plt.subplots()
    figsetting(measure, fig, ax)


#label the ID
    xxx=testset.ID
    yyy=[]
    zzz=[]
    sss=testset.category

    dopant_test = np.zeros((nTest,nFeat))
    dopant_test = XX.values

    n=0
    mm=0

    for i in range(nTest-1):
    
        if dopant_test[i][0] == dopant_test[i+1][0] and dopant_test[i][1] == dopant_test[i+1][1] and dopant_test[i][2] == dopant_test[i+1][2]           and dopant_test[i][3] == dopant_test[i+1][3]:
            yyy.append(Y_test_in[i])
            zzz.append(Y_fitted_org[i])
        
        else:
            if n > 10:
                n -= 11
                mm += 1
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.show()
                fig, ax = plt.subplots()
                figsetting(measure, fig, ax)
        
        #colors = mcd.CSS4_COLORS[overlap[2*n+1]]
            colors = mcd.CSS4_COLORS[color_array[n]]
            n +=1
            mm += 1
                
            plt.scatter(zzz,yyy,alpha=1.0, c=colors, edgecolors='black',label=writeid(xxx[i],dopant_test[i]))
            yyy=[]
            zzz=[]
        
            yyy.append(Y_train_in[i])
            zzz.append(Y_train_out_org[i])        
        

    yyy.append(Y_test_in[nTest-1])
    zzz.append(Y_fitted_org[nTest-1])        
    colors = mcd.CSS4_COLORS[color_array[n]]
    plt.scatter(zzz,yyy,alpha=1.0, c=colors, edgecolors='black',label=writeid(xxx[i],dopant_test[nTest-1])) 
    print('number of test samples:',mm)        
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    rmse = np.sqrt(mean_squared_error(Y_test_in, Y_fitted_org))
    print("RMSE: %.4f" % rmse)




    for i in range(nTest):
        if Y_test_in[i]*Y_fitted_org[i] < 0:
        
            print('+-',xxx[i], writeid(xxx[i],dopant_test[i]))
        elif np.abs(Y_test_in[i] - Y_fitted_org[i]) > rmse*2.0:
            print('>2',xxx[i], writeid(xxx[i],dopant_test[i]))

            
            
    return

