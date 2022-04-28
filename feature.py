import numpy as np
import math
from math import exp
from elements import ELEMENTS
import pandas as pd
from myfunction import linear
from myfunction import root
from myfunction import vbm_ordering
from myfunction import vbm_ordering_energy
from myfunction import cbm_ordering
from myfunction import cbm_ordering_energy




def construct_feature(X,calnum,features, training):
    mendeleev  = pd.read_csv('./inputdata/mendeleev.csv', index_col=0)
    info=len(features)+1 -8    #basic information #ordering subtract
    numspieces = 6   # Sn, Se. Na, dopant
    nFeat=numspieces+info+1
    
    #concentration, Band gap, VBM1_E, VBM1_M,CBM1_E,CBM1_M
    pristine=[0.5329, 5.3731, -0.1301, 5.9059, 0.0890,5.3403,5.2493,6.0760,6.1073,-0.1373,-0.2572,31.5707,0.1090]    
    se_vac=[0.4514, 5.5667, -0.4805, 6.0181, 0.1943,5.3935,5.3718,6.0725,6.1472,-0.5800,-0.9511,1.9001,1.2897 ]
    sn_vac=[0.6739, 5.3698, -0.1614, 6.0540, 0.1222,5.3801,5.3197,6.1058,6.2353,-0.1659, -0.3528,1.8322,0.5700]
    Na_3 = [0.7010, 5.3590, -0.1545, 6.0884, 0.1192,5.3818,5.3131,6.0828,6.2810,-0.1662,-0.4925,4.7726,0.1990]
    Cl_3 = [0.6602, 5.2816, -0.1697, 5.9983, 0.1029,5.3380,5.2800,6.0001,6.1504,-0.1602,-0.3007,7.1846,0.3120]

    pristine_top=[5.3731,-0.1301,5.9059,0.0890]
    se_vac_top=[5.5667,-0.4805,6.0181,0.1943]
    sn_vac_top=[5.3801,-0.1659,6.0540,0.1222]
    Na_3_top=[5.3818,-0.1662,6.0828,4.7726]
    Cl_3_top=[5.3380,-0.1602,5.9983,0.1029]


    
    initial=0
    initial1=1
    cal_ratio=0.03125
    
    nTrain=X.shape[0]
    X_train = np.zeros((nTrain,nFeat))
    
    dopant1       = X.dopant1_num.values
    dopant1_ratio = X.dopant1_ratio.values
    dopant2       = X.dopant2_num.values
    dopant2_ratio = X.dopant2_ratio.values
    
                                          
    #dopant 1 = Na or Cl or none, dopant2 = X
    for i in range(0,nTrain):
        if dopant1[i].astype(int) == 11:
            
            X_train[i,3+info] = dopant1_ratio[i]
            X_train[i,5+info] = dopant2[i]
            X_train[i,6+info] = dopant2_ratio[i]
            
            if (X_train[i,3+info] > 0.01) and (X_train[i,5+info] == 0):
                X_train[i,6+info] = X_train[i,3+info] - 0.01
                X_train[i,3+info] = 0.01
                X_train[i,5+info] = 11
                #print(X_train[i,3+info], X_train[i,6+info])

            
        elif  (dopant1[i].astype(int) == 17): 
            X_train[i,4+info] = dopant1_ratio[i]
            X_train[i,5+info] = dopant2[i]
            X_train[i,6+info] = dopant2_ratio[i]
           
            if (X_train[i,4+info] > 0.01) and (X_train[i,5+info] == 0):
                X_train[i,6+info] = X_train[i,4+info] - 0.01
                X_train[i,4+info] = 0.01
                X_train[i,5+info] = 17
                
        elif dopant2[i].astype(int) == 11:
           
            X_train[i,3+info] = dopant2_ratio[i]
            X_train[i,5+info] = dopant1[i]
            X_train[i,6+info] = dopant1_ratio[i]
           

        elif dopant2[i].astype(int) == 17:
            
            X_train[i,4+info] = dopant2_ratio[i]
            X_train[i,5+info] = dopant1[i]
            X_train[i,6+info] = dopant1_ratio[i]
                        
            
        else :
            X_train[i,5+info] = dopant1[i]
            X_train[i,6+info] = dopant1_ratio[i]
    
    # Sn, Se ratio, direction, temperature
    X_train[:,1+info] = X.Sn_ratio[:]
    X_train[:,2+info] = X.Se_ratio[:]    
    
    X_train[:,0] = X.direction.values    #measure direction
    X_train[:,1] = X.Temp.values         #temperature
    
   
    n=2
    for kk in features:
        

        if kk == 'vbm_e_sn':
            
            X_train[:,n] = linear(initial ,pristine[1],  cal_ratio, sn_vac[1], 1-X_train[:,1+info])

            vbm_e_sn=n
            n+=1
        elif kk == 'vbm_e2_sn':
            
            X_train[:,n] = linear(initial ,pristine[5],  cal_ratio, sn_vac[5], 1-X_train[:,1+info])

            vbm_e2_sn=n
            n+=1
        elif kk == 'vbm_e3_sn':
            
            X_train[:,n] = linear(initial ,pristine[6],  cal_ratio, sn_vac[6], 1-X_train[:,1+info])

            vbm_e3_sn=n
            n+=1            
            
        elif kk == 'vbm_e_se':
            
            X_train[:,n] = linear(initial ,pristine[1],  cal_ratio, se_vac[1], 1-X_train[:,2+info])

            vbm_e_se=n
            n+=1
            
        elif kk == 'vbm_e2_se':
            
            X_train[:,n] = linear(initial ,pristine[5],  cal_ratio, se_vac[5], 1-X_train[:,2+info])

            vbm_e2_se=n
            n+=1
            
        elif kk == 'vbm_e3_se':
            
            X_train[:,n] = linear(initial ,pristine[6],  cal_ratio, se_vac[6], 1-X_train[:,2+info])

            vbm_e3_se=n
            n+=1       
            
            
            
        elif kk == 'cbm_e_sn':
            
            X_train[:,n] = linear(initial ,pristine[3],  cal_ratio, sn_vac[3], 1-X_train[:,1+info])

            cbm_e_sn=n
            n+=1

        elif kk == 'cbm_e2_sn':
            
            X_train[:,n] = linear(initial ,pristine[7],  cal_ratio, sn_vac[7], 1-X_train[:,1+info])

            cbm_e2_sn=n
            n+=1
        elif kk == 'cbm_e3_sn':
            
            X_train[:,n] = linear(initial ,pristine[8],  cal_ratio, sn_vac[8], 1-X_train[:,1+info])

            cbm_e3_sn=n
            n+=1                
            
        elif kk == 'cbm_e_se':
            
            X_train[:,n] = linear(initial ,pristine[3],  cal_ratio, se_vac[3], 1-X_train[:,2+info])

            cbm_e_se=n
            n+=1
        elif kk == 'cbm_e2_se':
            
            X_train[:,n] = linear(initial ,pristine[7],  cal_ratio, se_vac[7], 1-X_train[:,2+info])

            cbm_e2_se=n
            n+=1
        elif kk == 'cbm_e3_se':
            
            X_train[:,n] = linear(initial ,pristine[8],  cal_ratio, se_vac[8], 1-X_train[:,2+info])

            cbm_e3_se=n
            n+=1       
            
        elif kk == 'vbm_e':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[1],  cal_ratio, calnum[X_train[i,5+info].astype(int),3],X_train[i,6+info]) +\
                                   linear(initial ,pristine[1],  cal_ratio, sn_vac[1], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[1],  cal_ratio, Na_3[1], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[1],  cal_ratio, calnum[X_train[i,5+info].astype(int),3],X_train[i,6+info]) + \
                                   linear(initial ,pristine[1],  cal_ratio, sn_vac[1], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[1],  cal_ratio, Cl_3[1], X_train[i,4+info])
                    
                else:
                    X_train[i,n] = linear(initial ,pristine[1],  cal_ratio, calnum[X_train[i,5+info].astype(int),3],X_train[i,6+info]) +\
                                   linear(initial ,pristine[1],  cal_ratio, sn_vac[1], 1-X_train[i,1+info]) - \
                                   pristine[1] 
                    

                    
            vbm_e=n    
            n+=1

        elif kk == 'vbm_e2':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[5],  cal_ratio, calnum[X_train[i,5+info].astype(int),5],X_train[i,6+info]) +\
                                   linear(initial ,pristine[5],  cal_ratio, sn_vac[5], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[5],  cal_ratio, Na_3[5], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[5],  cal_ratio, calnum[X_train[i,5+info].astype(int),5],X_train[i,6+info]) + \
                                   linear(initial ,pristine[5],  cal_ratio, sn_vac[5], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[5],  cal_ratio, Cl_3[5], X_train[i,4+info])
                    

                    
                else:
                    X_train[i,n] = linear(initial ,pristine[5],  cal_ratio, calnum[X_train[i,5+info].astype(int),5],X_train[i,6+info]) +\
                                   linear(initial ,pristine[5],  cal_ratio, sn_vac[5], 1-X_train[i,1+info]) - \
                                   pristine[5] 
                    

                    
            vbm_e2=n    
            n+=1
            
        elif kk == 'vbm_e3':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[6],  cal_ratio, calnum[X_train[i,5+info].astype(int),7],X_train[i,6+info]) +\
                                   linear(initial ,pristine[6],  cal_ratio, sn_vac[6], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[6],  cal_ratio, Na_3[6], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[6],  cal_ratio, calnum[X_train[i,5+info].astype(int),7],X_train[i,6+info]) + \
                                   linear(initial ,pristine[6],  cal_ratio, sn_vac[6], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[6],  cal_ratio, Cl_3[6], X_train[i,4+info])


                    
                else:
                    X_train[i,n] = linear(initial ,pristine[6],  cal_ratio, calnum[X_train[i,5+info].astype(int),7],X_train[i,6+info]) +\
                                   linear(initial ,pristine[6],  cal_ratio, sn_vac[6], 1-X_train[i,1+info]) - \
                                   pristine[6] #5.0978145
                    

                    
            vbm_e3=n    
            n+=1

        elif kk == 'cbm_e':
            
            for i in range(0,nTrain):
 
                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[3],  cal_ratio, calnum[X_train[i,5+info].astype(int),9],X_train[i,6+info]) + \
                                   linear(initial ,pristine[3],  cal_ratio, sn_vac[3], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[3],  cal_ratio, Na_3[3], X_train[i,3+info])

                                   
                    
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[3],  cal_ratio, calnum[X_train[i,5+info].astype(int),9],X_train[i,6+info]) + \
                                   linear(initial ,pristine[3],  cal_ratio, sn_vac[3], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[3],  cal_ratio, Cl_3[3], X_train[i,4+info])

                    
                else:
                    X_train[i,n] = linear(initial ,pristine[3],  cal_ratio, calnum[X_train[i,5+info].astype(int),9],X_train[i,6+info]) +\
                                   linear(initial ,pristine[3],  cal_ratio, sn_vac[3], 1-X_train[i,1+info]) - \
                                   pristine[3] #5.0978145            

                    
                          
            cbm_e = n              
            n+=1
        elif kk == 'cbm_e2':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[7],  cal_ratio, calnum[X_train[i,5+info].astype(int),11],X_train[i,6+info]) +\
                                   linear(initial ,pristine[7],  cal_ratio, sn_vac[7], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[7],  cal_ratio, Na_3[7], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[7],  cal_ratio, calnum[X_train[i,5+info].astype(int),11],X_train[i,6+info]) + \
                                   linear(initial ,pristine[7],  cal_ratio, sn_vac[7], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[7],  cal_ratio, Cl_3[7], X_train[i,4+info])
                    

                    
                else:
                    X_train[i,n] = linear(initial ,pristine[7],  cal_ratio, calnum[X_train[i,5+info].astype(int),11],X_train[i,6+info]) +\
                                   linear(initial ,pristine[7],  cal_ratio, sn_vac[7], 1-X_train[i,1+info]) - \
                                   pristine[7]
                    

                    
            cbm_e2=n    
            n+=1
            
        elif kk == 'cbm_e3':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[8],  cal_ratio, calnum[X_train[i,5+info].astype(int),13],X_train[i,6+info]) +\
                                   linear(initial ,pristine[8],  cal_ratio, sn_vac[8], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[8],  cal_ratio, Na_3[8], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine[8],  cal_ratio, calnum[X_train[i,5+info].astype(int),13],X_train[i,6+info]) + \
                                   linear(initial ,pristine[8],  cal_ratio, sn_vac[8], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine[8],  cal_ratio, Cl_3[8], X_train[i,4+info])
                    

                    
                else:
                    X_train[i,n] = linear(initial ,pristine[8],  cal_ratio, calnum[X_train[i,5+info].astype(int),13],X_train[i,6+info]) +\
                                   linear(initial ,pristine[8],  cal_ratio, sn_vac[8], 1-X_train[i,1+info]) - \
                                   pristine[8] 
                    

                    
            cbm_e3=n    
            n+=1
                  
                
        elif kk == 'vbm_org_1':
            X_train[:,n] = linear(initial ,pristine[1],  cal_ratio, calnum[X_train[:,5+info].astype(int),3],X_train[:,6+info])
            vbm_org_1=n
            n+=1
            
        elif kk == 'vbm_org_2':
            X_train[:,n] = linear(initial ,pristine[5],  cal_ratio, calnum[X_train[:,5+info].astype(int),5],X_train[:,6+info])
            vbm_org_2=n
            n+=1
            
        elif kk == 'vbm_org_3':
            X_train[:,n] = linear(initial ,pristine[6],  cal_ratio, calnum[X_train[:,5+info].astype(int),7],X_train[:,6+info])
            vbm_org_3=n
            n+=1            
            
        elif kk == 'cbm_org_1':
            X_train[:,n] = linear(initial ,pristine[3],  cal_ratio, calnum[X_train[:,5+info].astype(int),9],X_train[:,6+info])
            cbm_org_1=n
            n+=1
            
        elif kk == 'cbm_org_2':
            X_train[:,n] = linear(initial ,pristine[7],  cal_ratio, calnum[X_train[:,5+info].astype(int),11],X_train[:,6+info])
            cbm_org_2=n
            n+=1
            
        elif kk == 'cbm_org_3':
            X_train[:,n] = linear(initial ,pristine[8],  cal_ratio, calnum[X_train[:,5+info].astype(int),13],X_train[:,6+info]) 
            cbm_org_3=n
            n+=1            
            
            
            
        elif kk == 'vbm_e_top':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine_top[0],  cal_ratio, calnum[X_train[i,5+info].astype(int),17],X_train[i,6+info]) +\
                                   linear(initial ,pristine_top[0],  cal_ratio, sn_vac_top[0], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine_top[0],  cal_ratio, Na_3_top[0], X_train[i,3+info])
                    

                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine_top[0],  cal_ratio, calnum[X_train[i,5+info].astype(int),17],X_train[i,6+info]) + \
                                   linear(initial ,pristine_top[0],  cal_ratio, sn_vac_top[0], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine_top[0],  cal_ratio, Cl_3_top[0], X_train[i,4+info])
                    

                    
                else:
                    X_train[i,n] = linear(initial ,pristine_top[0],  cal_ratio, calnum[X_train[i,5+info].astype(int),17],X_train[i,6+info]) +\
                                   linear(initial ,pristine_top[0],  cal_ratio, sn_vac_top[0], 1-X_train[i,1+info]) - \
                                   pristine_top[0] 
                
                
            vbm_e_top=n    
            n+=1     

        elif kk == 'vbm_m_top':
            
            X_train[:,n] = linear(initial ,pristine_top[1],  cal_ratio, calnum[X_train[:,5+info].astype(int),18],X_train[:,6+info])
                     
                     
            vbm_m_top=n
            n+=1    
            
        elif kk == 'cbm_e_top':
            for i in range(0,nTrain):

                if X_train[i,3+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine_top[2],  cal_ratio, calnum[X_train[i,5+info].astype(int),19],X_train[i,6+info]) +\
                                   linear(initial ,pristine_top[2],  cal_ratio, sn_vac_top[2], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine_top[2],  cal_ratio, Na_3_top[2], X_train[i,3+info])


                                   
                elif X_train[i,4+info] != 0 : 
                    X_train[i,n] = linear(initial ,pristine_top[2],  cal_ratio, calnum[X_train[i,5+info].astype(int),19],X_train[i,6+info]) + \
                                   linear(initial ,pristine_top[2],  cal_ratio, sn_vac_top[2], 1-X_train[i,1+info]) - \
                                   linear(initial ,pristine_top[2],  cal_ratio, Cl_3_top[2], X_train[i,4+info])
                    

                    
                else:
                    X_train[i,n] = linear(initial ,pristine_top[2],  cal_ratio, calnum[X_train[i,5+info].astype(int),19],X_train[i,6+info]) +\
                                   linear(initial ,pristine_top[2],  cal_ratio, sn_vac_top[2], 1-X_train[i,1+info]) - \
                                   pristine_top[2] #5.0978145
                    

                    
            cbm_e_top=n    
            n+=1     

        elif kk == 'cbm_m_top':
            
            X_train[:,n] = linear(initial ,pristine_top[3],  cal_ratio, calnum[X_train[:,5+info].astype(int),20],X_train[:,6+info])
                     
                     
            cbm_m_top=n
            n+=1             
        elif kk == 'vbm_m':
            
            X_train[:,n] = linear(initial ,pristine[2],  cal_ratio, calnum[X_train[:,5+info].astype(int),4],X_train[:,6+info])
                     
                     
            vbm_m=n
            n+=1

        elif kk == 'vbm_m_sn':
            
            X_train[:,n] = linear(initial ,pristine[2],  cal_ratio, sn_vac[2], 1-X_train[:,1+info])

            vbm_m_sn=n
            n+=1            

        elif kk == 'vbm_m_se':
            
            X_train[:,n] = linear(initial ,pristine[2],  cal_ratio, se_vac[2], 1-X_train[:,2+info]) 
            vbm_m_se=n
            n+=1            

        elif kk == 'vbm_m2_sn':
            
            X_train[:,n] = linear(initial ,pristine[9],  cal_ratio, sn_vac[9], 1-X_train[:,1+info])

            vbm_m2_sn=n
            n+=1            

        elif kk == 'vbm_m2_se':
            
            X_train[:,n] = linear(initial ,pristine[9],  cal_ratio, se_vac[9], 1-X_train[:,2+info]) 
            vbm_m2_se=n
            n+=1            
                        
        elif kk == 'vbm_m3_sn':
            
            X_train[:,n] = linear(initial ,pristine[10],  cal_ratio, sn_vac[10], 1-X_train[:,1+info])

            vbm_m3_sn=n
            n+=1            

        elif kk == 'vbm_m3_se':
            
            X_train[:,n] = linear(initial ,pristine[10],  cal_ratio, se_vac[10], 1-X_train[:,2+info]) 
            vbm_m3_se=n
            n+=1            
                                                    
        elif kk == 'vbm_m2':
            
            X_train[:,n] = linear(initial ,pristine[9],  cal_ratio, calnum[X_train[:,5+info].astype(int),6],X_train[:,6+info])
                     
                     
            vbm_m2=n
            n+=1

        elif kk == 'vbm_m3':
            
            X_train[:,n] = linear(initial ,pristine[10],  cal_ratio, calnum[X_train[:,5+info].astype(int),8],X_train[:,6+info])
                     
                     
            vbm_m3=n
            n+=1

         
        elif kk == 'cbm_m':
            
            X_train[:,n] = linear(initial ,pristine[4],  cal_ratio, calnum[X_train[:,5+info].astype(int),10],X_train[:,6+info])
                       
                           #linear(initial ,pristine[4],  cal_ratio, se_vac[4], 1-X_train[:,2+info]) 
            cbm_m=n
            n+=1       

        elif kk == 'cbm_m2':
            
            X_train[:,n] = linear(initial ,pristine[11],  cal_ratio, calnum[X_train[:,5+info].astype(int),12],X_train[:,6+info])
                       
                           #linear(initial ,pristine[4],  cal_ratio, se_vac[4], 1-X_train[:,2+info]) 
            cbm_m2=n
            n+=1       

        elif kk == 'cbm_m3':
          
            X_train[:,n] = linear(initial ,pristine[12],  cal_ratio, calnum[X_train[:,5+info].astype(int),14],X_train[:,6+info])
                       
                           #linear(initial ,pristine[4],  cal_ratio, se_vac[4], 1-X_train[:,2+info]) 
            cbm_m3=n
            n+=1       
            
        elif kk == 'cbm_m_sn':
            
            X_train[:,n] = linear(initial ,pristine[4],  cal_ratio, sn_vac[4], 1-X_train[:,1+info]) 
                           
            cbm_m_sn=n
            n+=1  

        elif kk == 'cbm_m_se':
            
            X_train[:,n] = linear(initial ,pristine[4],  cal_ratio, se_vac[4], 1-X_train[:,2+info]) 
            cbm_m_se=n
            n+=1  
            
        elif kk == 'cbm_m2_sn':
            
            X_train[:,n] = linear(initial ,pristine[11],  cal_ratio, sn_vac[11], 1-X_train[:,1+info]) 
                           
            cbm_m2_sn=n
            n+=1  

        elif kk == 'cbm_m2_se':
            
            X_train[:,n] = linear(initial ,pristine[11],  cal_ratio, se_vac[11], 1-X_train[:,2+info]) 
            cbm_m2_se=n
            n+=1  
        elif kk == 'cbm_m3_sn':
            
            X_train[:,n] = linear(initial ,pristine[12],  cal_ratio, sn_vac[12], 1-X_train[:,1+info]) 
                           
            cbm_m3_sn=n
            n+=1  

        elif kk == 'cbm_m3_se':
            
            X_train[:,n] = linear(initial ,pristine[12],  cal_ratio, se_vac[12], 1-X_train[:,2+info]) 
            cbm_m3_se=n
            n+=1  
                                         
        elif kk == 'magnetism':
            X_train[:,n] = calnum[dopant1.astype(int),15] + calnum[dopant2.astype(int),15]
           #print('magnetism',X_train[:,n])
            n+=1
            
        elif kk == 'deep_level':
            X_train[:,n] = calnum[dopant1.astype(int),16] + calnum[dopant2.astype(int),16]
            n+=1
            
        elif kk == 'vbm_ordering':
            
            X_train=vbm_ordering_energy(X_train,vbm_e,vbm_e2,vbm_e3)
            
            
        elif kk == 'cbm_ordering':
            
            X_train=cbm_ordering_energy(X_train,cbm_e,cbm_e2,cbm_e3)

        elif kk == 'vbm_org_ordering':
            
            X_train=vbm_ordering(X_train,vbm_org_1,vbm_org_2,vbm_org_3,vbm_m,vbm_m2,vbm_m3)
                              

        elif kk == 'cbm_org_ordering':
            
            X_train=cbm_ordering(X_train,cbm_org_1,cbm_org_2,cbm_org_3,cbm_m,cbm_m2,cbm_m3)
            
        elif kk == 'vbm_sn_ordering':
            
            X_train=vbm_ordering(X_train,vbm_e_sn,vbm_e2_sn,vbm_e3_sn,vbm_m_sn,vbm_m2_sn,vbm_m3_sn)
            
            
        elif kk == 'cbm_sn_ordering':
            
            X_train=cbm_ordering(X_train,cbm_e_sn,cbm_e2_sn,cbm_e3_sn,cbm_m_sn,cbm_m2_sn,cbm_m3_sn)            
            
        elif kk == 'vbm_se_ordering':
            
            X_train=vbm_ordering(X_train,vbm_e_se,vbm_e2_se,vbm_e3_se,vbm_m_se,vbm_m2_se,vbm_m3_se)
            
            
        elif kk == 'cbm_se_ordering':
            
            X_train=cbm_ordering(X_train,cbm_e_se,cbm_e2_se,cbm_e3_se,cbm_m_se,cbm_m2_se,cbm_m3_se)            
            
        elif kk == 'bandgap':

            X_train[:,n] = X_train[:,cbm_org_1] - X_train[:,vbm_org_1]
                    
            bandgap = n
                           # bandgap
            n+=1
            

  
        if kk == 'bandgap_sn':
            
            X_train[:,n] = X_train[:,cbm_e_sn] - X_train[:,vbm_e_sn]

            bandgap_sn=n
            n+=1

        elif kk == 'bandgap_se':
            
            X_train[:,n] = X_train[:,cbm_e_se] - X_train[:,cbm_e_se]

            bandgap_se=n
            n+=1
            
            
        elif kk == 'bandgap_rate':

            X_train[:,n] = X_train[:,bandgap]*X_train[:,6+info]+\
                           X_train[:,bandgap_sn]*(1-X_train[:,1+info]) +\
                           X_train[:,bandgap_se]*(1-X_train[:,2+info])
            bandgap_rate = n
                           # bandgap
            n+=1            
                        
        elif kk == 'ion':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]   
            
            ion=n
            n+=1
            
          
        elif kk == 'covalent':

            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]     
            

            covalent=n
            n+=1


        elif kk == 'vdw':

            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
                        

            vdw=n
            n+=1            
            
            
        elif kk == 'atomic_radius':

            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]    
            
            atomic_rad=n
            n+=1   
            
            

        elif kk == 'pauling_en':

            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  

            pauling_en=n
            n+=1                 

        elif kk == 'en_allen':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
                
            en_allen_n=n
            n+=1      
      
        elif kk == 'polarizability':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
            polarizability_n=n
            n+=1                 
     
                        
        elif kk == 'melting_point':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
            melting=n
            n+=1              
      
                        
        elif kk == 'boiling_point':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
            boiling=n
            n+=1                 


                        
        elif kk == 'thermal_conductivity':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
            thermal=n
            n+=1                 


        elif kk == 'specific_heat':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
                  
            
            specific=n
            n+=1    

                        
        elif kk == 'density':
            
            temp1 = np.array(mendeleev.iloc[dopant1.astype(int)][kk])
            temp2 = np.array(mendeleev.iloc[dopant2.astype(int)][kk])
            temp3 = mendeleev.iloc[50][kk]
            temp4 = mendeleev.iloc[34][kk]            

            value=[temp1,temp2,temp3,temp4]
            ratio=[dopant1_ratio,dopant2_ratio,X_train[:,1+info], X_train[:,2+info]]
            
                        
            for i in range(len(value)):
                X_train[:,n] += value[i]*ratio[i]  
            density_n=n
            n+=1          
       
    #attach ID list
    ids=X.ID.values
    column=ids.reshape(-1,1)
    X_train = np.hstack((column,X_train))    
    

    #attach ZT values
    if training == True:
        target=X.K.values
        column2=target.reshape(-1,1)

  
        X_train = np.hstack((X_train,column2))

    
    return X_train


