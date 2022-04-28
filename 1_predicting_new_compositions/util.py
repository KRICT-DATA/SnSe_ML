
import numpy
import pandas
import random
import itertools
import matplotlib.pyplot as plt
from joblib import dump, load
import xgboost as xgb
from elements import ELEMENTS
seed=200

class KFold_seed:
    def __init__(self, data_dict, num_feats, num_folds, seed, normalization=True):
        self.data_dict = data_dict
        self.num_feats = num_feats
        self.num_folds = num_folds
        self.seed = seed
        self.normalization = normalization
        self.folds = list()
        self.ids = list(self.data_dict.keys())

        #self.split_dataset()

    def split_dataset(self):
        id_list=list()
        num_ids = len(self.ids)
        size_fold = int(num_ids / self.num_folds)

        random.seed(self.seed)
        random.shuffle(self.ids)
        for k in range(0, self.num_folds - 1):
            self.folds.append(self.ids[k*size_fold:(k+1)*size_fold])

        self.folds.append(self.ids[(self.num_folds-1)*size_fold:])

        return self.folds

    def split_dataset_zt(self):
        id_list=list()
        num_ids = len(self.ids)
        size_fold = int(num_ids / self.num_folds)

        for k in range(0, self.num_folds - 1):
            self.folds.append(self.ids[k*size_fold:(k+1)*size_fold])

        self.folds.append(self.ids[(self.num_folds-1)*size_fold:])

        return self.folds
    
    def get(self, k):
        train_ids = list(itertools.chain.from_iterable([x for i, x in enumerate(self.folds) if i != k]))
        test_ids = self.folds[k]
        train_data = list()
        test_data = list()

        for id in train_ids:
            for data in self.data_dict[id]:
                train_data.append(data)
        train_data = numpy.vstack(train_data)
        if self.normalization:
            train_data, train_mean, train_std = zscore2(train_data)

        for id in test_ids:
            for data in self.data_dict[id]:
                test_data.append(data)
        test_data = numpy.vstack(test_data)
        if self.normalization:
            test_data = zscore(test_data, train_mean, train_std)

        return train_data, test_data

    def get_zt(self,id):
        
        test_data = list()
        
        for data in self.data_dict[id]:
            test_data.append(data)
        test_data = numpy.vstack(test_data)
        
        return test_data


class KFold:
    def __init__(self, data_dict, num_feats, num_folds, normalization=True):
        self.data_dict = data_dict
        self.num_feats = num_feats
        self.num_folds = num_folds
        self.normalization = normalization
        self.folds = list()
        self.ids = list(self.data_dict.keys())

        self.split_dataset()

    def split_dataset(self):
        num_ids = len(self.ids)
        size_fold = int(num_ids / self.num_folds)

        random.seed(54)
        random.shuffle(self.ids)
        for k in range(0, self.num_folds - 1):
            self.folds.append(self.ids[k*size_fold:(k+1)*size_fold])

        self.folds.append(self.ids[(self.num_folds-1)*size_fold:])

    def get(self, k):
        train_ids = list(itertools.chain.from_iterable([x for i, x in enumerate(self.folds) if i != k]))
        test_ids = self.folds[k]
        train_data = list()
        test_data = list()

        for id in train_ids:
            for data in self.data_dict[id]:
                train_data.append(data)
        train_data = numpy.vstack(train_data)
        if self.normalization:
            train_data, train_mean, train_std = zscore2(train_data)

        for id in test_ids:
            for data in self.data_dict[id]:
                test_data.append(data)
        test_data = numpy.vstack(test_data)
        if self.normalization:
            test_data = zscore(test_data, train_mean, train_std)

        return train_data, test_data

def load_data(file_name):
    data = numpy.array(pandas.read_csv(file_name))
    data_dict = dict()
    for i in range(0, data.shape[0]):
        if data[i, 1] not in data_dict.keys():
            data_dict[data[i, 1]] = list()
        data_dict[data[i, 1]].append(numpy.array(data[i, 2:], dtype=numpy.float))

    return data_dict


def zscore(X, means, stds):

    for i in range(0, X.shape[0]):
        for j in range(0,X.shape[1]-1):
            if stds[j] == 0:
                    X[i,j] = 0
            else:
                    X[i,j] = (X[i,j] - means[j]) / stds[j]

    return X


def zscore2(X):
    means = numpy.mean(X, axis=0)
    stds = numpy.std(X, axis=0)

    for i in range(0, X.shape[0]):
        for j in range(0,X.shape[1]-1):
            if stds[j] == 0:
                    X[i,j] = 0
            else:
                    X[i,j] = (X[i,j] - means[j]) / stds[j]

    return X, means, stds



def figsetting(measure, fig, ax):
    if measure == 'thermal_conductivity':

        ax.set_xlim((0, 1.6))
        ax.set_ylim((0, 1.6))
        ax.set_xlabel('measured thermal conductivity')
        ax.set_ylabel('predicted thermal conductivity')

    elif measure == 'electrical_conductivity':

        ax.set_xlim((0, 180))
        ax.set_ylim((0, 180))
        ax.set_xlabel('measured electrical conductivity')
        ax.set_ylabel('predicted electrical conductivity')

    elif measure == 'seebeck_coeff':

        ax.set_xlim((-600, 700))
        ax.set_ylim((-600, 700))
        ax.set_xlabel('measured seebeck coefficient')
        ax.set_ylabel('predicted seebeck coefficient')

    x = numpy.linspace(-20000, 20000, 10)
    ax.plot(x, x, color='black')
    ax.set_aspect('equal')
    return

def draw_feature_importance(filename, column):
    
    model=load(filename)
    num_feats=len(column)
    
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = numpy.argsort(feature_importance)
    temp=[]
    for i in range(num_feats):
        temp.append(column[sorted_idx[i]])
    print(temp)
    temp=[]
    for i in range(num_feats):
        temp.append(feature_importance[sorted_idx[i]])
    print(temp)
    
    cutoff=20
    if num_feats > cutoff:
        sorted_idx2=numpy.zeros((cutoff,), dtype=int)
        for ii in range(cutoff):
            sorted_idx2[cutoff-ii-1] = sorted_idx[sorted_idx.shape[0]-ii-1]
        pos = numpy.arange(sorted_idx2.shape[0]) + .5

        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx2], align='center')

        temp=[]
        for j in range(cutoff):
            
            temp.append(column[sorted_idx2[j]])
    
    else :
        pos = numpy.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        temp=[]
        for j in range(num_feats):
            temp.append(column[sorted_idx[j]])    
    plt.yticks(pos,(temp))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show() 
    return

     
def dopant_list(thermo):
    nTest1=thermo.shape[0] 
    dopant1=thermo.dopant1_num
    dopant2=thermo.dopant2_num
    dopant1_ratio = thermo.dopant1_ratio
    dopant2_ratio = thermo.dopant2_ratio
    X_train=numpy.zeros((nTest1,4))    

    
    for i in range(0,nTest1):
        if dopant1[i].astype(int) == 11:
            X_train[i,0] = 11    
            X_train[i,1] = dopant1_ratio[i]
            X_train[i,2] = dopant2[i]
            X_train[i,3] = dopant2_ratio[i]


        elif  (dopant1[i].astype(int) == 17): 
            X_train[i,0] = 17    
            X_train[i,1] = dopant1_ratio[i]
            X_train[i,2] = dopant2[i]
            X_train[i,3] = dopant2_ratio[i]

        elif dopant2[i].astype(int) == 11:
            X_train[i,0] = 11    
            X_train[i,1] = dopant2_ratio[i]
            X_train[i,2] = dopant1[i]
            X_train[i,3] = dopant1_ratio[i]


        elif dopant2[i].astype(int) == 17:

            X_train[i,0] = 17    
            X_train[i,1] = dopant2_ratio[i]
            X_train[i,2] = dopant1[i]
            X_train[i,3] = dopant1_ratio[i]


        else :
            X_train[i,2] = dopant1[i]
            X_train[i,3] = dopant1_ratio[i]    
            
    return(X_train)    


def scatterplot(target,ZT,dopant2_name,label_name,file_pdf):
    fig = plt.figure(figsize=(20,20))
    fig, ax = plt.subplots()
    ax.set_xlim((0,2.0))
    ax.set_ylim((0,2.0))
    ax.set_xlabel('measured ZT')
    ax.set_ylabel('predicted ZT')



    x = numpy.linspace(-15000, 15000, 10)
    ax.plot(x,x,color='black')
    ax.set_aspect('equal')


    plt.scatter(target,ZT,alpha=0.4,c=dopant2_name,edgecolors='black', cmap=plt.cm.rainbow) #
    plt.colorbar(label=label_name)
    plt.show()
    #ax.set_rasterized(True)
    fig.savefig(file_pdf, format='pdf')
    
    return

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
def temp_plot(temp,target,ZT,file_pdf):
    
    target1=[]
    target2=[]
    target3=[]
    target4=[]
    target5=[]
    target6=[]

    ZT1=[]
    ZT2=[]
    ZT3=[]
    ZT4=[]
    ZT5=[]
    ZT6=[]

    for i in range(len(ZT)):
        if temp[i] <= 400.0 :
            target1.append(target[i])
            ZT1.append(ZT[i])
        elif temp[i] <= 500.0 : 
            target2.append(target[i])
            ZT2.append(ZT[i])        
        elif temp[i] <= 600.0 : 
            target3.append(target[i])
            ZT3.append(ZT[i])        
        elif temp[i] <= 700.0 : 
            target4.append(target[i])
            ZT4.append(ZT[i])        
        else : 
            target5.append(target[i])
            ZT5.append(ZT[i])        
        #elif temp[i] <= 900.0 : 
        #    target6.append(target[i])
        #    ZT6.append(ZT[i])        

    target_sum=[target1,target2,target3,target4,target5]
    ZT_sum=[ZT1,ZT2,ZT3,ZT4,ZT5]

    xxx=[]
    yyy=[]
    mae=[]
    r2score=[]
    for xx, yy in zip(target_sum,ZT_sum):
        rmse_train = numpy.sqrt(mean_squared_error(xx, yy))
        mae_train = mean_absolute_error(xx,yy)
        r2score_train = r2_score(yy,xx)
        #print("%.4f" % rmse_train,"%.4f" % mae_train,"%.4f" % r2score_train)
        xxx.extend(xx)
        yyy.extend(yy)
        mae.append(mae_train)
        r2score.append(r2score_train)


    #print(numpy.sqrt(mean_squared_error(xxx, yyy)), mean_absolute_error(xxx,yyy), r2_score(yyy,xxx))


    label = ['300K-400K', '400K-500K', '500K-600K', '600K-700K', '>700K']

    index = numpy.arange(len(label))

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = fig.add_subplot(111, sharex=ax, frameon=False)

    ax.set_xlabel('temperature range')
    width=0.35
    ax.bar(index, mae, width, color='red')
    ax2.bar(index+width, r2score, width, color='blue')

    ax2.set_xticks([], [])
    ax.set_xlim(-0.5,len(index)-0.1)
    ax2.set_xlim(-0.5,len(index)-0.1)
    ax.set_ylim((0,0.18))
    ax2.set_ylim((0,1))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax.set_ylabel('MAE')
    ax2.set_ylabel('R2 score')
    ax.set_xticks(index+width/2)
    xtickNames = ax.set_xticklabels(label)
    plt.setp(xtickNames)



    plt.show()

    fig.savefig(file_pdf, format='pdf')    
    return

import copy
def cut_feature(column2,feature, test_data_x):
    
    temp = copy.deepcopy(column2)
    temp2=[]
#    for j in feature[iii]:
    for j in feature:
        temp.remove(j)
                
    for k in temp:
        temp2.append(column2.index(k))
    temp2.sort(reverse=True)    
    
    testin = copy.deepcopy(test_data_x) 
    
    #remove useless feature columns
    for j in temp2:
        testin  = numpy.delete(testin,j,axis=1) 
        
    pandas.DataFrame(testin).to_csv('test_features_nofeatures.csv')
    
    return testin

def calculate_ZT(predicted,temp,target, thermo,logical,file_name):
    nTest=thermo.shape[0]
    ZT = predicted[1,:]*predicted[1,:] * predicted[0,:]*temp[:]/ predicted[2,:] *10**(-10)

    elcond_pd=[]
    seebeck_pd=[]
    thermal_pd=[]
    ZT_pd=[]
    abs_pd=[]
    for i in range(nTest):
        elcond_pd.append(predicted[0,i])
        seebeck_pd.append(predicted[1,i])
        thermal_pd.append(predicted[2,i])
        ZT_pd.append(ZT[i])
        if logical == True:
            abs_pd.append(numpy.abs(ZT[i]-target[i]))


    thermo.loc[:, 'el_cond'] = pandas.Series(elcond_pd, index=thermo.index)
    thermo.loc[:, 'seebeck'] = pandas.Series(seebeck_pd, index=thermo.index)
    thermo.loc[:, 'thermal'] = pandas.Series(thermal_pd, index=thermo.index)
    thermo.loc[:, 'ZT'] = pandas.Series(ZT_pd, index=thermo.index)
    if logical == True:
        thermo.loc[:, 'abs'] = pandas.Series(abs_pd, index=thermo.index)
    #display(thermo)
    thermo.to_csv(file_name)
    
    return ZT


def pred_train_ML(model,train_data_x_in,train_dataset,total_train_in, total_train_out, num_feats):
    
    pred_train = model.predict(train_data_x_in)
    total_train_in.extend(train_dataset[:,num_feats])
    total_train_out.extend(pred_train)
    train_mae = numpy.mean(numpy.abs(train_dataset[:,num_feats] - pred_train))
    train_r2 = r2_score(pred_train,train_dataset[:,num_feats])

    return train_mae, train_r2
    
def pred_test_ML(model,num_feats, test_data_x_in,test_data_y, total_test_in, total_test_out):    
    pred_test = model.predict(test_data_x_in)
    total_test_in.extend(test_data_y)
    total_test_out.extend(pred_test)
    test_mae = numpy.mean(numpy.abs(test_data_y - pred_test))
    test_r2 = r2_score(pred_test, test_data_y)   
    
    return test_mae, test_r2
    
def draw_pickup(thermo, ZT, elcond_pd, seebeck_pd, thermal_pd, doping):

    temp_pick=[]
    zt_pick=[]
    dopant_pick=[]
    id_pick=[]

    id_pick_1=[]
    id_pick_2=[]
    el_pick=[]
    seebeck_pick=[]
    thermal_pick=[]

    id=thermo.ID
    testttt=ZT.sort_values(ascending=False)
    temp=thermo.Temp    


    n=0

    
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,2)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4) 
    
    ax0.set_xlim((300,820))
    ax1.set_xlim((300,820))
    ax2.set_xlim((300,820))
    ax3.set_xlim((300,820))
    
    ax0.set_xlabel('Temperature (K)')
    ax1.set_xlabel('Temperature (K)')
    ax2.set_xlabel('Temperature (K)')
    ax3.set_xlabel('Temperature (K)')
    
    ax0.set_ylim((0,3))
    ax1.set_ylim((0,150))
    ax2.set_ylim((0,600))
    if seebeck_pd[0] < 0.0:
        ax1.set_ylim((0,-600))
    
    ax3.set_ylim((0,1.5))
    
    
    ax0.set_ylabel('predicted ZT')
    ax1.set_ylabel('electrical conductivity (S/cm)')
    ax2.set_ylabel('Seebeck coefficient (muV/K)')
    ax3.set_ylabel('thermal conductivity (W/mK)')
    
    for i in range(len(temp)):

        temp_pick.append(temp[i])
        zt_pick.append(ZT[i])
        el_pick.append(elcond_pd[i])
        seebeck_pick.append(seebeck_pd[i])
        thermal_pick.append(thermal_pd[i])
        
        if i == len(temp)-1:
            ax0.plot(temp_pick,zt_pick,marker="o", label=id[i])
            ax1.plot(temp_pick,el_pick,marker="o", label=id[i])
            ax2.plot(temp_pick,seebeck_pick,marker="o", label=id[i])
            ax3.plot(temp_pick,thermal_pick,marker="o", label=id[i])
                
        elif id[i] != id[i+1]:
            ax0.plot(temp_pick,zt_pick,marker="o", label=id[i])
            ax1.plot(temp_pick,el_pick,marker="o", label=id[i])
            ax2.plot(temp_pick,seebeck_pick,marker="o", label=id[i])
            ax3.plot(temp_pick,thermal_pick,marker="o", label=id[i])
                
                #print(id[i], np.max(zt_pick))
            temp_pick=[]
            zt_pick=[]
            el_pick=[]
            seebeck_pick=[]
            thermal_pick=[]





    plt.legend(loc='upper right')
    plt.show()
    #ax.set_rasterized(True)

    return


def dopantremove(drop_num, chosen,ordering):
    measureset=['electrical_conductivity','seebeck_coeff','thermal_conductivity']

    for measure in measureset:


        #training info
        file = str(measure)+'_1_total.csv'
        thermo  = pandas.read_csv(file, index_col=0)

        remove = thermo[(thermo['dopant2_num'] != drop_num) & (thermo['dopant1_num'] != drop_num) ]
        remove.reset_index(inplace=True, drop=True)
        #del remove['index']
        pandas.DataFrame(remove).to_csv(str(measure)+'_1_total_dopantremove.csv')
        #print(remove)

        #training features
        load_file = 'training_features_'+str(measure)+'_total_'+str(ordering)+'.csv'
        data_dict = pandas.read_csv(load_file, index_col=0)

        new_feature = data_dict[data_dict['dopant_num'] != drop_num]
        new_feature.reset_index(inplace=True, drop=True)
        pandas.DataFrame(new_feature).to_csv('training_features_'+str(measure)+'_total_dopantremove.csv')


    #ZT test info
    zt_info  = pandas.read_csv('zt_dataset_lowtemp.csv', index_col=0)
    survive = zt_info[(zt_info['dopant2_num'] == drop_num) | (zt_info['dopant1_num'] == drop_num)]
    survive.reset_index(inplace=True, drop=True)
    pandas.DataFrame(survive).to_csv('zt_dataset_lowtemp_dopantremove.csv')

    zt_feature = pandas.read_csv('zt_test_features_'+str(ordering)+'.csv', index_col=0)
    survive = zt_feature[zt_feature['dopant_num'] == drop_num]
    survive.reset_index(inplace=True, drop=True)
    pandas.DataFrame(survive).to_csv('zt_test_features_dopantremove.csv')    


    for measure in measureset:


        #training info
        file = str(measure)+'_1_total.csv'
        thermo  = pandas.read_csv(file, index_col=0)

        remove = thermo[(thermo['dopant2_num'] != drop_num) & (thermo['dopant1_num'] != drop_num) ]
        remove = remove.append(thermo[(thermo['ID']==chosen)])
        remove.reset_index(inplace=True, drop=True)

        pandas.DataFrame(remove).to_csv(str(measure)+'_1_total_dopantremove_exist.csv')
        #print(remove)

        #training features
        load_file = 'training_features_'+str(measure)+'_total_'+str(ordering)+'.csv'
        data_dict = pandas.read_csv(load_file, index_col=0)

        new_feature = data_dict[data_dict['dopant_num'] != drop_num]
        new_feature = new_feature.append(data_dict[(thermo['ID']==chosen)])
        new_feature.reset_index(inplace=True, drop=True)
        pandas.DataFrame(new_feature).to_csv('training_features_'+str(measure)+'_total_dopantremove_exist.csv')


    #ZT test info
    zt_info  = pandas.read_csv('zt_dataset_lowtemp.csv', index_col=0)
    
    survive = zt_info[(zt_info['dopant2_num'] == drop_num) | (zt_info['dopant1_num'] == drop_num)]
    survive = survive[(survive['ID'] != chosen)]
    survive.reset_index(inplace=True, drop=True)
    pandas.DataFrame(survive).to_csv('zt_dataset_lowtemp_dopantremove_exist.csv')

    zt_feature = pandas.read_csv('zt_test_features_'+str(ordering)+'.csv', index_col=0)
    survive = zt_feature[zt_feature['dopant_num'] == drop_num]
    survive = survive[(survive['id'] != chosen)]
    survive.reset_index(inplace=True, drop=True)
    pandas.DataFrame(survive).to_csv('zt_test_features_dopantremove_exist.csv')     
    
    return

def define_category(composition):
        
    category1=0
    if composition[4] == 11:
        category1 = 1
    elif composition[4] == 17:
        category1 = 2


    composition.append(category1)
    
    return composition

