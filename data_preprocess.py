#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:38:32 2018

@author: chendongcai
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import knn_impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from fancyimpute import KNN
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
import seaborn as sns
from scipy.stats import boxcox
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
#%%

rootdir='/Files/predict poverty/data'
all_file=os.listdir(rootdir)
all_file=sorted(all_file)
data_pool=[]
for i in range(len(all_file)):
    path=os.path.join(rootdir,all_file[i])
    locals()[all_file[i][:-4]]=pd.read_csv(path)
    data_pool.append(locals()[all_file[i][:-4]])
    
household_train_index=[2,6,10]
household_test_index=[1,5,9]
indiv_train_index=[4,8,12]
indiv_test_index=[3,7,11]

missing_index=[]
for i in range(1,13):
    count_missing=(data_pool[i].isnull().sum()/data_pool[i].shape[0]).sort_values(ascending=False)
    count=count_missing.values.sum()
    if count!=0:
        missing_index.append(i)
        
# data_pool[3] OdXpbPGJ  missing ratio:0.165255
# 0.78559482060965735 的 values 是 4
# data_pool[4] OdXpbPGJ  missing ratio:0.16688
# 0.78370607028753991 的 values 是 4
# data_pool[5] 
#IrxBnWxE    0.913342
#dnlnKrAg    0.847880
#FGWqGkmD    0.802369
#aAufyreG    0.733167
#umkFMfvA    0.725062 delete
#BRzuVmyf    0.450748
#McFBIGsm    0.225062
#BXOWgPgL    0.225062
#OSmfjCbE    0.223815  
#data_pool[6]
#IrxBnWxE    0.916436
#dnlnKrAg    0.836559
#FGWqGkmD    0.815054
#umkFMfvA    0.726575
#aAufyreG    0.720737 delete
#BRzuVmyf    0.448848
#OSmfjCbE    0.230722
#McFBIGsm    0.230722
#BXOWgPgL    0.230722
#data_pool[7]
#AJgudnHB    0.998112
#DYgxQeEi    0.997417
#sIiSADFG    0.997218
#jfsTwowc    0.994635
#hdDTwJhQ    0.993741
#WmKLEUcd    0.984006
#nxAFXxLQ    0.984006
#MGfpfHam    0.979833
#uDmhgsaQ    0.979833
#HZqPmvkr    0.979833
#DtcKwIEv    0.951719
#ETgxnJOM    0.951619
#qlLzyqpP    0.943771
#fyfDnyQk    0.922710
#unRAgFtX    0.922710
#sWElQwuC    0.922710
#iZhWxnWa    0.905226
#tzYvQeOb    0.899563
#NfpXxGQk    0.816710
#WqEZQuJP    0.745778
#CLTXEwmz    0.732962
#DSttkpSI    0.732962
#BoxViLPz    0.732962 delete
#jzBRbsEG    0.476356
#TJGiunYp    0.381085
#esHWAAyG    0.377210
#TZDgOhYY    0.349891
#mAeaImix    0.185377
#data_pool[8]
#AJgudnHB    0.998963
#DYgxQeEi    0.997630
#sIiSADFG    0.996692
#hdDTwJhQ    0.995112
#jfsTwowc    0.993285
#WmKLEUcd    0.982520
#nxAFXxLQ    0.982175
#HZqPmvkr    0.978373
#MGfpfHam    0.978373
#uDmhgsaQ    0.978373
#DtcKwIEv    0.950968
#ETgxnJOM    0.950474
#qlLzyqpP    0.941487
#fyfDnyQk    0.918428
#unRAgFtX    0.918428
#sWElQwuC    0.918378
#tzYvQeOb    0.900010
#iZhWxnWa    0.894233
#NfpXxGQk    0.812167
#WqEZQuJP    0.734051
#BoxViLPz    0.730446
#DSttkpSI    0.730446
#CLTXEwmz    0.730397 delete
#jzBRbsEG    0.486915
#TJGiunYp    0.385048
#esHWAAyG    0.367766
#TZDgOhYY    0.341448
#mAeaImix    0.184081
#%%
#new_data=data_pool
#for i in missing_index:
#    count_missing=(data_pool[i].isnull().sum()/data_pool[i].shape[0]).sort_values(ascending=False)
#    new_data[i]=new_data[i].drop(list(count_missing.index[count_missing>0.5]),axis=1)
#
#for i in missing_index:
#    count_missing=(new_data[i].isnull().sum()/new_data[i].shape[0]).sort_values(ascending=False)
#    print(count_missing)
    
new_data[4][new_data[4].dtypes[new_data[4].dtypes!='object'].index]
#OdXpbPGJ ukWqmeSS are 2 numerical columns
new_data[4].set_index(['id','iid'])
    
def compress(data):    
    df={}
    for i in data.columns:
        df[i]=[]
    index=list(data.id.unique())
    for i in index:
        try:
            temp_series=data.loc[data.id==i,:].apply(lambda x:list(set(x)),axis=0)
        except:
            print('Error')
        for j in data.columns:
            if type(temp_series) is not pd.core.frame.DataFrame:
                df[j].append(temp_series[j])
            else:
                df[j].append(list(temp_series[j].values))
    df=pd.DataFrame(df)
#    df.poor=df.poor.apply(lambda x: x[0])
    df.id=df.id.apply(lambda x: x[0])
    df['family_member']=df.iid.apply(lambda x:len(x))
    return df

#%%
def df_to_dict(df):
    dicti={}
    for i in df.columns:
        dicti[i]=[]
    for i in df.columns:
        dicti[i]=list(df[i].values)
    return dicti

def compress_data(data):    
    df={}
    for i in data.columns:
        df[i]=[]
    index=list(data.id.unique())
    for i in index:
        try:
            temp_series=df_to_dict(data.loc[data.id==i,:])
        except:
            print('Error')
        for j in data.columns:
            df[j].append(temp_series[j])
    df=pd.DataFrame(df)
#    df.poor=df.poor.apply(lambda x: x[0])
    df.id=df.id.apply(lambda x: x[0])
    df['family_member']=df.iid.apply(lambda x:len(x))
    num_idx=list(data.dtypes[data.dtypes!='object'].index)
    num_idx.extend(['family_member','country'])
    counter=df.drop(num_idx,axis=1).applymap(lambda x:Counter(x))
    new_dicti={}
    for i in df.drop(num_idx,axis=1).columns:
        for j in data[i].unique():
            new_dicti[j]=[]
    for i in df.drop(num_idx,axis=1).columns:
        for j in counter[i]:
            for k in data[i].unique():
                new_dicti[k].append(j.get(k,0))
    result=pd.DataFrame(new_dicti)
    return df,result   

def create_data(indi_train,indi_test,hhold_train,hhold_test):
    d_train1,d_train2=compress_data(indi_train)
    d_test1,d_test2=compress_data(indi_test)
    num_idx=list(indi_train.dtypes[indi_train.dtypes!='object'].index)
    num_idx.extend(['family_member','country'])
    indi_train_trans=pd.merge(d_train1.loc[:,num_idx],d_train2,right_index=True,left_index=True)  
    indi_train_trans=indi_train_trans.drop('iid',axis=1)
    indi_train_trans.country=indi_train_trans.country.apply(lambda x:list(x)[0])
    indi_train_trans.poor=indi_train_trans.poor.apply(lambda x:list(x)[0])
#    A_train.ukWqmeSS=A_train.ukWqmeSS.apply(lambda x:np.mean(np.array(x)))
#    A_train.OdXpbPGJ=A_train.OdXpbPGJ.apply(lambda x:np.median(np.array(x)))
    
#    hhold=hhold_train
    hhold_num=hhold_train[hhold_train.dtypes[hhold_train.dtypes!='object'].index]
    hhold_cate=hhold_train[hhold_train.dtypes[hhold_train.dtypes=='object'].index]
    new_hhold_cate=pd.get_dummies(hhold_cate)
    new_hhold=pd.merge(new_hhold_cate,hhold_num,left_index=True,right_index=True)
    for i in indi_train_trans.columns:
        if i not in num_idx:
            indi_train_trans[i]=indi_train_trans[i]/indi_train_trans['family_member']
    indi_train_trans=indi_train_trans.drop('poor',axis=1)
    trans_train_data=pd.merge(new_hhold,indi_train_trans,left_on='id',right_on='id')
# test  
    
    num_idx_test=list(indi_test.dtypes[indi_test.dtypes!='object'].index)
    num_idx_test.extend(['family_member','country'])
    indi_test_trans=pd.merge(d_test1.loc[:,num_idx_test],d_test2,right_index=True,left_index=True)  
    indi_test_trans=indi_test_trans.drop('iid',axis=1)
    indi_test_trans.country=indi_test_trans.country.apply(lambda x:list(x)[0])
#    A_test.ukWqmeSS=A_test.ukWqmeSS.apply(lambda x:np.mean(np.array(x)))
#    A_test.OdXpbPGJ=A_test.OdXpbPGJ.apply(lambda x:np.median(np.array(x)))

#    hhold_t=hhold_test
    hhold_t_num=hhold_test[hhold_test.dtypes[hhold_test.dtypes!='object'].index]
    hhold_t_cate=hhold_test[hhold_test.dtypes[hhold_test.dtypes=='object'].index]
    new_hhold_t_cate=pd.get_dummies(hhold_t_cate)
    new_hhold_t=pd.merge(new_hhold_t_cate,hhold_t_num,left_index=True,right_index=True)

    for i in indi_test_trans.columns:
        if i not in num_idx_test:
            indi_test_trans[i]=indi_test_trans[i]/indi_test_trans['family_member']
#A_train=A_train.drop('poor',axis=1)
    trans_test_data=pd.merge(new_hhold_t,indi_test_trans,left_on='id',right_on='id')

    target=trans_train_data['poor']
    union_set=set(trans_train_data.columns).intersection(set(trans_test_data.columns))
    drop_test=list(set(trans_test_data.columns).difference(union_set))
    drop_train=list(set(trans_train_data.columns).difference(union_set))
    trans_test_data=trans_test_data.drop(drop_test,axis=1)
    trans_train_data=trans_train_data.drop(drop_train,axis=1)

    all_data=pd.concat([trans_train_data,trans_test_data],axis=0)
    train_index=trans_train_data.shape[0]
    return all_data,train_index

def normalize(data,columns):
    for i in columns:
        mean=data[i].mean()
        std=data[i].std()
        data[i]=(data[i]-mean)/std
    return data

#score,f=feature_selection_chi2(nd2,600)
#f=pd.DataFrame(f)
#f['poor']=A_train_data['poor']
#f['country']=A_train_data['country']
#test_2(f)
#    print('mean log loss:', ll)
#check corr pls
def feature_selection_chi2(data,target,k_val):
    test = SelectKBest(score_func=chi2, k=k_val)
    Y=target
#   X=data  
    X=data
    fit = test.fit(X, Y)
# summarize scores
    np.set_printoptions(precision=3)
    features = fit.transform(X)
    return fit.scores_,features

def LassoLR(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.15)
#    lrcv=LogisticRegressionCV(Cs=[0.01,0.1,0.5,1,5,10,20],fit_intercept=True,cv=10,penalty='l2',scoring=None, tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv=LogisticRegressionCV(Cs=[0.01,0.1,0.2,0.4,0.5,0.6],fit_intercept=True,cv=10,penalty='l1',scoring=None,solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv.fit(X_train,y_train)
    y_pred_train=lrcv.predict_proba(X_test)
    y_pred_label=lrcv.predict(X_test)
    acc=np.mean(y_pred_label==y_test)
    ll=log_loss(y_test,y_pred_train)
    y_pred_test=lrcv.predict_proba(data.iloc[train_index:,:])
    return ll,lrcv.C_,y_pred_test,acc,y_pred_train

def RF(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.2)
    rfc=RandomForestClassifier(criterion='entropy',max_features=200,max_depth=10,n_estimators=300)
    rfc.fit(X_train,y_train)
    y_pred=rfc.predict_proba(X_test)
    ll=log_loss(y_test,y_pred)
    return ll,rfc.feature_importances_

def svm_linear(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    clf=svm.SVC(C=1,probability=True,kernel='linear')
    clf.fit(X_train,y_train)
    y_pred=clf.predict_proba(X_test)
    ll=log_loss(y_test,y_pred)
    return ll

def NN(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    model=Sequential()
    model.add(Dense(1000,input_shape=(data.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='adam')
    y_train=np_utils.to_categorical(y_train,2)
    y_test=np_utils.to_categorical(y_test,2)
    model.fit(X_train, y_train,
          batch_size=50, nb_epoch=3,
          verbose=1,
          validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test,verbose=0)
    y_pred=model.predict_proba(X_test, batch_size=50)
    print(y_pred)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

B_raw_data=create_data(data_pool[8],data_pool[7],data_pool[6],data_pool[9])
B_raw_data[0].to_csv('/Files/predict poverty/B_raw_data_all.csv')
