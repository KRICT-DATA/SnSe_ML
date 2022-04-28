import numpy as np
import math
from math import exp
from elements import ELEMENTS
from myfunction import linear
from mendeleev import element

add=5    #+3+3+4  # 5: calculation, 3: exp(-E), 3:exp(-E/T) 3: E^0.5, E^-0.5,E^1.5, E^-1.5
caladd=1+add*3+ 1 + add*2    # 1:mass, 10:temp, add:snvac, add: sevac
numspieces = 42


#Xnum index
snratio      =0 ; seratio      =1 ; dopant1      =2; dopant1_ratio=3 
dopant2      =4 ; dopant2_ratio=5 ; dopant3      =6; dopant3_ratio=7
direction    =8 ; Temp         =9 ; categorycal  =10; expinvT     =11

bandgap      =12 ; vbm1_e       =13; vbm1_m       =14
cbm1_e       =15 ; cbm1_m       =16

#expe_g       =16 ; expe_v       =17 ; expe_c       =18
#expegT       =19 ; expevT       =20 ; expecT       =21
#eg_12        =22 ; eg_m12       =23 ; eg_32       =24 ; eg_m32 = 25

mass         = expinvT + add*3 +1

#temp_12 = mass + 1 ; temp_m12 = mass+2 
#temp_32 = mass + 3 ; temp_m32 = mass+4
#temp_52 = mass + 5 ; temp_m52 = mass+6
#temp_2  = mass + 7 ; temp_m2  = mass+8
#temp_3  = mass + 9 ; temp_m3  = mass+10

temp_m3 = mass

snvac_g  = temp_m3 + 1 ;
snvac_v  = temp_m3 + 2 ;
snvac_vm = temp_m3 + 3 ;
snvac_c  = temp_m3 + 4 ;
snvac_cm = temp_m3 + 5 ;

#exp_snvac_g  = temp_m3 + 6 ;
#exp_snvac_v  = temp_m3 + 7 ;
#exp_snvac_c  = temp_m3 + 8 ;#

#exp_snvac_gT = temp_m3 + 9 ;
#exp_snvac_vT = temp_m3 + 10 ;
#exp_snvac_cT = temp_m3 + 11;#

#exp_snvac_eg_12 = temp_m3 + 12;
#exp_snvac_eg_m12 = temp_m3 + 13;
#exp_snvac_eg_32 = temp_m3 + 14;
#exp_snvac_eg_m32 = temp_m3 + 15;


sevac_g  = temp_m3 + 6 ;
sevac_v  = temp_m3 + 7 ;
sevac_vm = temp_m3 + 8 ;
sevac_c  = temp_m3 + 9 ;
sevac_cm = temp_m3 + 10 ;

#exp_sevac_g  = temp_m3 + 21 ;
#exp_sevac_v  = temp_m3 + 22 ;
#exp_sevac_c  = temp_m3 + 23 ;

#exp_sevac_gT = temp_m3 + 24 ;
#exp_sevac_vT = temp_m3 + 25 ;
#exp_sevac_cT = temp_m3 + 26;

#exp_sevac_eg_12 = temp_m3 + 27;
#exp_sevac_eg_m12 = temp_m3 + 28;
#exp_sevac_eg_32 = temp_m3 + 29;
#exp_sevac_eg_m32 = temp_m3 + 30;

#exp_snvac_vm = temp_m3 + 14 ;
#exp_snvac_cm = temp_m3 + 15 ;
#exp_sevac_vm = temp_m3 + 19 ;
#exp_sevac_cm = temp_m3 + 20 ;

#cal index
cal_band     =6 ; cal_vbme     =7 ; cal_vbmm     =8  
cal_cbme     =9 ; cal_cbmm     =10

cal_ratio    =0.03125     #theoretical data set, ratio all fixed
initial      =0



def feature_engineer(Xnum,calnum,number,pristine, sn_vac, se_vac,ii,end):
 
    temp=[]
    
    if number == 1:
        dopant = dopant1
        dopant_ratio = dopant1_ratio
        do2 = 0
        
    elif number == 2:
        dopant = dopant2
        dopant_ratio = dopant2_ratio
        do2 = add
        
    elif number == 3:
        dopant = dopant3
        dopant_ratio = dopant3_ratio
        do2 = add*2
        

    Xnum[mass] += ELEMENTS[Xnum[dopant]].mass * Xnum[dopant_ratio]
    Xnum[bandgap+do2]= linear(initial,pristine[0],cal_ratio, calnum[cal_band],Xnum[dopant_ratio])
    Xnum[vbm1_e+do2] = linear(initial,pristine[1],cal_ratio, calnum[cal_vbme],Xnum[dopant_ratio])
    Xnum[vbm1_m+do2] = linear(initial,pristine[2],cal_ratio, calnum[cal_vbmm],Xnum[dopant_ratio])
    Xnum[cbm1_e+do2] = linear(initial,pristine[3],cal_ratio, calnum[cal_cbme],Xnum[dopant_ratio])
    Xnum[cbm1_m+do2] = linear(initial,pristine[4],cal_ratio, calnum[cal_cbmm],Xnum[dopant_ratio])
    
 #   Xnum[expe_g+do2] = exp(-Xnum[bandgap+do2])
 #   Xnum[expe_v+do2] = exp(-Xnum[vbm1_e+do2])
 #   Xnum[expe_c+do2] = exp(-Xnum[cbm1_e+do2])
 #   Xnum[expegT+do2] = Xnum[expe_g+do2] * Xnum[Temp]
 #   Xnum[expevT+do2] = Xnum[expe_v+do2] * Xnum[Temp]
 #   Xnum[expecT+do2] = Xnum[expe_c+do2] * Xnum[Temp]
 #   Xnum[eg_12+do2]  = abs(Xnum[bandgap+do2])**0.5
 #   Xnum[eg_m12+do2] = abs(Xnum[bandgap+do2])**(-0.5)
 #   Xnum[eg_32+do2]  = abs(Xnum[bandgap+do2])**(float(3)/2)
 #   Xnum[eg_m32+do2] = abs(Xnum[bandgap+do2])**(-float(3)/2)   

    Xnum[end+ii] = Xnum[dopant_ratio]
    
    return Xnum

def get_calvalues(Xnum_row,calnum,pristine, sn_vac, se_vac, measurement,end):

    initial1=1
    
    if measurement == 'thermal_conductivity':
        Xnum_row[exp_sevac_eg_m32+numspieces+1] = Xnum_row[Temp+1]   #density
    Xnum_row[expinvT]  = exp(-1/Xnum_row[Temp])        
    Xnum_row[mass] = ELEMENTS['Sn'].mass*Xnum_row[snratio]+ ELEMENTS['Se'].mass*Xnum_row[seratio]

    #Xnum_row[temp_12]  = Xnum_row[Temp]**0.5
    #Xnum_row[temp_m12] = Xnum_row[Temp]**(-0.5)
    #Xnum_row[temp_32]  = Xnum_row[Temp]**(1.5)
    #Xnum_row[temp_m32] = Xnum_row[Temp]**(-1.5)
    #Xnum_row[temp_52]  = Xnum_row[Temp]**(2.5)
    #Xnum_row[temp_m52] = Xnum_row[Temp]**(-2.5)
    #Xnum_row[temp_2]   = Xnum_row[Temp]**2.0
    #Xnum_row[temp_m2]  = Xnum_row[Temp]**(-2.0)
    #Xnum_row[temp_3]   = Xnum_row[Temp]**3.0
    #Xnum_row[temp_m3]  = Xnum_row[Temp]**(-3.0)
        
    Xnum_row[snvac_g]  = linear(initial1,pristine[0],1-cal_ratio, sn_vac[0], Xnum_row[snratio])
    Xnum_row[snvac_v]  = linear(initial1,pristine[1],1-cal_ratio, sn_vac[1], Xnum_row[snratio])
    Xnum_row[snvac_vm] = linear(initial1,pristine[2],1-cal_ratio, sn_vac[2], Xnum_row[snratio])
    Xnum_row[snvac_c]  = linear(initial1,pristine[3],1-cal_ratio, sn_vac[3], Xnum_row[snratio])
    Xnum_row[snvac_cm] = linear(initial1,pristine[4],1-cal_ratio, sn_vac[4], Xnum_row[snratio])

   # Xnum_row[exp_snvac_g]  = exp(-Xnum_row[snvac_g])
   # Xnum_row[exp_snvac_v]  = exp(-Xnum_row[snvac_v])
   # Xnum_row[exp_snvac_c]  = exp(-Xnum_row[snvac_c])

    #Xnum_row[exp_snvac_gT] = Xnum_row[exp_snvac_g] * Xnum_row[Temp]
    #Xnum_row[exp_snvac_vT] = Xnum_row[exp_snvac_v] * Xnum_row[Temp]
    #Xnum_row[exp_snvac_cT] = Xnum_row[exp_snvac_c] * Xnum_row[Temp]

    #Xnum_row[exp_snvac_eg_12]  = abs(Xnum_row[exp_snvac_g])**0.5
    #Xnum_row[exp_snvac_eg_m12] = abs(Xnum_row[exp_snvac_g])**(-0.5)
    #Xnum_row[exp_snvac_eg_32]  = abs(Xnum_row[exp_snvac_g])**(1.5)
    #Xnum_row[exp_snvac_eg_m32] = abs(Xnum_row[exp_snvac_g])**(-1.5)   
    
    
    Xnum_row[sevac_g]  = linear(initial1,pristine[0],1-cal_ratio, se_vac[0], Xnum_row[seratio])
    Xnum_row[sevac_v]  = linear(initial1,pristine[1],1-cal_ratio, se_vac[1], Xnum_row[seratio])
    Xnum_row[sevac_vm] = linear(initial1,pristine[2],1-cal_ratio, se_vac[2], Xnum_row[seratio])
    Xnum_row[sevac_c]  = linear(initial1,pristine[3],1-cal_ratio, se_vac[3], Xnum_row[seratio])
    Xnum_row[sevac_cm] = linear(initial1,pristine[4],1-cal_ratio, se_vac[4], Xnum_row[seratio])
        
    #Xnum_row[exp_sevac_g]  = exp(-Xnum_row[sevac_g])
    #Xnum_row[exp_sevac_v]  = exp(-Xnum_row[sevac_v])
    #Xnum_row[exp_sevac_c]  = exp(-Xnum_row[sevac_c])     
    
    #Xnum_row[exp_sevac_gT] = Xnum_row[exp_sevac_g] * Xnum_row[Temp]
    #Xnum_row[exp_sevac_vT] = Xnum_row[exp_sevac_v] * Xnum_row[Temp]
    #Xnum_row[exp_sevac_cT] = Xnum_row[exp_sevac_c] * Xnum_row[Temp]
    #Xnum_row[exp_sevac_eg_12]  = abs(Xnum_row[exp_sevac_g])**0.5
    #Xnum_row[exp_sevac_eg_m12] = abs(Xnum_row[exp_sevac_g])**(-0.5)
    #Xnum_row[exp_sevac_eg_32]  = abs(Xnum_row[exp_sevac_g])**(1.5)
    #Xnum_row[exp_sevac_eg_m32] = abs(Xnum_row[exp_sevac_g])**(-1.5)   

    
  
    #Xnum_row[exp_snvac_vm] = exp(-Xnum_row[snvac_vm])
    #Xnum_row[exp_snvac_cm] = exp(-Xnum_row[snvac_cm])

    #Xnum_row[exp_sevac_vm] = exp(-Xnum_row[sevac_vm])        
    #Xnum_row[exp_sevac_cm] = exp(-Xnum_row[sevac_cm])        






    for ii in range(calnum.shape[0]):
          
        if Xnum_row[dopant1] == ELEMENTS[calnum[ii][0]].number :
            
            feature_engineer(Xnum_row,calnum[ii],1,pristine, sn_vac, se_vac,ii,end)
            
        
        elif Xnum_row[dopant2] == ELEMENTS[calnum[ii][0]].number :

            feature_engineer(Xnum_row,calnum[ii],2,pristine, sn_vac, se_vac,ii,end)

        elif Xnum_row[dopant3] == ELEMENTS[calnum[ii][0]].number :

            feature_engineer(Xnum_row,calnum[ii],3,pristine, sn_vac, se_vac,ii,end)


            
            
    #min, max, avg of calculation data
#    if Xnum_row[bandgap+add*2] == 0 :
            
#        if Xnum_row[bandgap+add] == 0 and Xnum_row[bandgap] == 0:
#            #print('no dopant')
#            for jjj in range(0,add):                
#                Xnum_row[bandgap+jjj]       = min(Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                Xnum_row[bandgap+jjj+add]   = max(Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                Xnum_row[bandgap+jjj+add*2] = (Xnum_row[snvac_g+jjj] + Xnum_row[sevac_g+jjj])/2.0
                
                
 #       elif Xnum_row[bandgap+add] == 0 and Xnum_row[bandgap] != 0 :
            
            #print('dopant #1')    
#            for jjj in range(0,add):
#                # # of dopants =1
#                temp1=min(Xnum_row[bandgap+jjj], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                temp2=max(Xnum_row[bandgap+jjj], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                temp3=(Xnum_row[bandgap+jjj] + Xnum_row[snvac_g+jjj] + Xnum_row[sevac_g+jjj])/3.0
#                
#                Xnum_row[bandgap+jjj]       = temp1
#                Xnum_row[bandgap+jjj+add]   = temp2
#                Xnum_row[bandgap+jjj+add*2] = temp3
                
            
#        else:
#            #print('dopant #2')    
#            for jjj in range(0,add):
#                # of dopants =2
#                #print(Xnum_row[bandgap+jjj],Xnum_row[bandgap+jjj+add], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                temp1=min(Xnum_row[bandgap+jjj],Xnum_row[bandgap+jjj+add], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                temp2=max(Xnum_row[bandgap+jjj],Xnum_row[bandgap+jjj+add], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#                temp3=(Xnum_row[bandgap+jjj] + Xnum_row[bandgap+jjj+add]+ Xnum_row[snvac_g+jjj]+ Xnum_row[sevac_g+jjj])/4.0
                
#                Xnum_row[bandgap+jjj] = temp1
#                Xnum_row[bandgap+jjj+add] = temp2
#                Xnum_row[bandgap+jjj+add*2] = temp3
        
#    else:              
#        #print('dopant #3')    

#        for jjj in range(0,add):            
#            # of dopants =3
#            temp1=min(Xnum_row[bandgap+jjj], Xnum_row[bandgap+jjj+add], Xnum_row[bandgap+jjj+add*2], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#            temp2=max(Xnum_row[bandgap+jjj], Xnum_row[bandgap+jjj+add], Xnum_row[bandgap+jjj+add*2], Xnum_row[snvac_g+jjj], Xnum_row[sevac_g+jjj])
#            temp3=   (Xnum_row[bandgap+jjj] +Xnum_row[bandgap+jjj+add] +Xnum_row[bandgap+jjj+add*2]+ Xnum_row[snvac_g+jjj]+ Xnum_row[sevac_g+jjj])/5.0
                
#            Xnum_row[bandgap+jjj] = temp1
#            Xnum_row[bandgap+jjj+add] = temp2
#            Xnum_row[bandgap+jjj+add*2] = temp3
            
                         
    #print(Xnum_row)
    
    return Xnum_row
                     

    
    
def construct_features(X,calnum,pristine,sn_vac,se_vac,measurement):
    
    # dimension
    #add=5 #+3+3+4  # 5: calculation, 3: exp(-E), 3:exp(-E/T) 3: E^0.5, E^-0.5,E^1.5, E^-1.5
    #caladd=1+add*3+ 1 + 10 + add*2 
    
    caladd2= caladd + numspieces  #dopant spiecies : 41

    ndim0=X.shape
    
    Xnum = np.zeros((ndim0[0],ndim0[1]+caladd2))
    end = ndim0[1] + caladd
    
    # copy values
    Xnum = X.values
    
     # add columns to Xnum
    c = np.zeros((ndim0[0]))

    for i in range(caladd2):
        Xnum=np.column_stack((Xnum,c))
    
    

    # evaluate features values
    for i in range(Xnum.shape[0]):
        get_calvalues(Xnum[i],calnum,pristine, sn_vac, se_vac, measurement,end)
        

    # remove the data of doping rate and one-hot encoding columns
    
    #print(Xnum.shape)
    #start=ndim0[1]+caladd
    #start = Xnum.shape[1] -numspieces-1
    #end = Xnum.shape[1] 
    #print('start',start)
    #for j in range(end-start+1):
    #    Xnum=np.delete(Xnum,[start],1)
    Xnum=np.delete(Xnum,[2,3,4,5,6,7,expinvT, bandgap, bandgap+add, bandgap+add*2,snvac_g, snvac_v, snvac_vm, snvac_c, snvac_cm, sevac_g,sevac_v,sevac_vm,sevac_c,sevac_cm,\
                         vbm1_e,vbm1_e+add,vbm1_e +add+add, vbm1_m, vbm1_m+add, vbm1_m+add+add,cbm1_e, cbm1_e+add,cbm1_e+add+add,\
                       cbm1_m, cbm1_m+add, cbm1_m+add+add ],1)
    #Xnum=np.delete(Xnum,[97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129.130,131,132,133,134,135,136,137,138,139,140,141,142],1)    
    
    print(Xnum.shape)
    return Xnum



def onehotencoder(X,calnum):
    info=5    #basic information
    numspieces = 83   # up to 'Bi'
    nFeat=numspieces+info+1
    #nFeat=info+1
    
    pristine=[0.429493,4.883068,-0.126278,5.312561,0.093205]   #concentration, Band gap, VBM1_E, VBM1_M,CBM1_E,CBM1_M
    sn_vac=[0.375528,4.908882,-0.160809,5.284410,0.118839]
    se_vac=[0.259513,5.047156,-0.256996,5.306669,0.223789]

    
    initial=0
    initial1=1
    
    nTrain=X.shape[0]
    X_train = np.zeros((nTrain,nFeat))
    eg=np.zeros((nTrain,5))

    dopant1=X.dopant1_num[0]
    dopant2=X.dopant2_num[0]
    dopant3=X.dopant3_num[0]

    X_train[0,dopant1+info] = X.dopant1_ratio[0]
    X_train[0,dopant2+info] = X.dopant2_ratio[0]
    X_train[0,dopant3+info] = X.dopant3_ratio[0]    
            
    X_train[0,50+info] = X.Sn_ratio[0]
    X_train[0,34+info] = X.Se_ratio[0]    
    
    
    if dopant3 != 0:
                
        temp1 = element(int(dopant1)).ionenergies[1]
        temp2 = element(int(dopant2)).ionenergies[1]
        temp3 = element(int(dopant3)).ionenergies[1]
                
        print(temp1,temp2,temp3)
        X_train[0,3] = np.min([temp1,temp2,temp3])
        X_train[0,4] = np.max([temp1,temp2,temp3])
        X_train[0,5] = np.average([temp1,temp2,temp3])
                
                #X_train[i,5] = element(int(dopant1)).electronegativity(scale='pauling')
                #if dopant1 == 12 or dopant1 == 25 or dopant1 == 80 or dopant1 == 30  or dopant1 == 48:
                #    X_train[i,5] = 0
    elif dopant2 != 0:

        temp1 = element(int(dopant1)).ionenergies[1]
        temp2 = element(int(dopant2)).ionenergies[1]

                
                #X_train[i,dopant1+info] = 1.0
        X_train[0,3] = np.min([temp1,temp2])
        X_train[0,4] = np.max([temp1,temp2])
        X_train[0,5] = np.average([temp1,temp2])
                
    elif dopant1 != 0:
                
        temp1 = element(int(dopant1)).ionenergies[1]
                
                #X_train[i,dopant1+info] = 1.0
        X_train[0,3] = temp1
        X_train[0,4] = temp1
        X_train[0,5] = temp1    
    
    
    
    #if dopant1 != 0:
        #X_train[0,dopant1+info] = 1.0
     #   X_train[0,3] = element(int(dopant1)).ionenergies[1]
        #X_train[0,5] = element(int(dopant1)).electronegativity(scale='pauling')
        #if dopant1 == 12 or dopant1 == 25 or dopant1 == 80 or dopant1 == 30  or dopant1 == 48:
        #    X_train[0,5] = 0
    #if dopant2 != 0:
        #X_train[0,dopant2+info] = 1.0
     #   X_train[0,4] = element(int(dopant2)).ionenergies[1]
        #X_train[0,6] = element(int(dopant2)).electronegativity(scale='pauling')
        #if dopant2 == 12 or dopant2 == 25 or dopant2 == 80 or dopant2 == 30 or dopant2 == 48:
        #    X_train[0,6] = 0        
    #if dopant3 != 0:
        #X_train[0,dopant3+info] = 1.0
    #    X_train[0,5] = element(int(dopant3)).ionenergies[1]
        #X_train[0,7] = element(int(dopant3)).electronegativity(scale='pauling')
        #if dopant3 == 12 or dopant3 == 25 or dopant3 == 80 or dopant3 == 30 or dopant3 == 48:
        #    X_train[0,7] = 0

    
    
    for i in range(1, nTrain):
        dopant1=X.dopant1_num[i]
        dopant2=X.dopant2_num[i]
        dopant3=X.dopant3_num[i]

        before1=X.dopant1_num[i-1]
        before2=X.dopant2_num[i-1]
        before3=X.dopant3_num[i-1]

        if dopant1 == before1 and dopant2 == before2 and dopant3 == before3 :
 
            X_train[i,:] = X_train[i-1,:]
        
        else:

            
            X_train[i,dopant1+info] = X.dopant1_ratio[i]
            X_train[i,dopant2+info] = X.dopant2_ratio[i]
            X_train[i,dopant3+info] = X.dopant3_ratio[i]    
            
            X_train[i,50+info] = X.Sn_ratio[i]
            X_train[i,34+info] = X.Se_ratio[i]
            
            if dopant3 != 0:
                
                temp1 = element(int(dopant1)).ionenergies[1]
                temp2 = element(int(dopant2)).ionenergies[1]
                temp3 = element(int(dopant3)).ionenergies[1]
                
                #X_train[i,dopant1+info] = 1.0
                X_train[i,3] = np.min([temp1,temp2,temp3])
                X_train[i,4] = np.max([temp1,temp2,temp3])
                X_train[i,5] = np.average([temp1,temp2,temp3])
                
                #X_train[i,5] = element(int(dopant1)).electronegativity(scale='pauling')
                #if dopant1 == 12 or dopant1 == 25 or dopant1 == 80 or dopant1 == 30  or dopant1 == 48:
                #    X_train[i,5] = 0
            elif dopant2 != 0:

                temp1 = element(int(dopant1)).ionenergies[1]
                temp2 = element(int(dopant2)).ionenergies[1]

                
                #X_train[i,dopant1+info] = 1.0
                X_train[i,3] = np.min([temp1,temp2])
                X_train[i,4] = np.max([temp1,temp2])
                X_train[i,5] = np.average([temp1,temp2])
                
                
                #X_train[i,4] = element(int(dopant2)).ionenergies[1]
                
                
                
                
                
                #X_train[i,dopant2+info] = 1.0
                
                #X_train[i,6] = element(int(dopant2)).electronegativity(scale='pauling')
                #if dopant2 == 12 or dopant2 == 25 or dopant2 == 80 or dopant2 == 30 or dopant2 == 48:
                #    X_train[i,6] = 0
            elif dopant1 != 0:
                
                temp1 = element(int(dopant1)).ionenergies[1]
                
                #X_train[i,dopant1+info] = 1.0
                X_train[i,3] = temp1
                X_train[i,4] = temp1
                X_train[i,5] = temp1
                
                
                
                
                #X_train[i,dopant3+info] = 1.0
                #X_train[i,5] = element(int(dopant3)).ionenergies[1]
                #X_train[i,7] = element(int(dopant3)).electronegativity(scale='pauling')
                #if dopant3 == 12 or dopant3 == 25 or dopant3 == 80 or dopant3 == 30 or dopant3 == 48:
                #    X_train[i,7] = 0      

            


                
    #X_train[:,0] = X.Sn_ratio.values
    #X_train[:,1] = X.Se_ratio.values
    X_train[:,0] = X.direction.values
    X_train[:,1] = X.category.values
    X_train[:,2] = X.Temp.values
    
    
    #for j in range(5):
    #    eg[:,0] = linear(initial1,pristine[0],1-cal_ratio, sn_vac[j],            X_train[:,0])
    #    eg[:,1] = linear(initial1,pristine[0],1-cal_ratio, se_vac[j],            X_train[:,1])
    #    eg[:,2] = linear(initial ,pristine[0],  cal_ratio, calnum[X.dopant1_num-1,j+6],X.dopant1_ratio)
    #    eg[:,3] = linear(initial ,pristine[0],  cal_ratio, calnum[X.dopant2_num-1,j+6],X.dopant2_ratio)
    #    eg[:,4] = linear(initial ,pristine[0],  cal_ratio, calnum[X.dopant3_num-1,j+6],X.dopant3_ratio)
    #    
    #    X_train[:,j] = np.average(eg,axis=1)
        #if j == 2:
        #    X_train[:,5] = np.max(eg, axis=1)
        #elif j==3:
        #    X_train[:,6] = np.min(eg, axis=1)
        
        
        #if j == 1 or j == 4:
            #X_train[:,j+5] = np.max(eg, axis=1)
        #else:
            #X_train[:,j+5] = np.min(eg, axis=1)
    #X_train=np.delete(X_train,[3],1)
    return X_train

def onehotencoder2(X,info,nFeat):
    nTrain=X.shape[0]
    X_train = np.zeros((nTrain,nFeat))


    for i in range(nTrain):
        dopant1=X.dopant1_num[i]
        dopant2=X.dopant2_num[i]
        dopant3=X.dopant3_num[i]
    
        X_train[i,dopant1+info] = X.dopant1_ratio[i]
        X_train[i,dopant2+info] = X.dopant2_ratio[i]
        X_train[i,dopant3+info] = X.dopant3_ratio[i]

    X_train[:,0] = X.Sn_ratio.values
    X_train[:,1] = X.Se_ratio.values
    X_train[:,2] = X.direction.values
    X_train[:,3] = X.K.values
    X_train[:,4] = X.Temp.values

    return X_train