#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:25:16 2018

@author: chendongcai
"""

import os
import re
import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
#import knn_impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
#from fancyimpute import KNN
#from collections import Counter
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif
from scipy.stats import kendalltau
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from pyfm import pylibfm
import pywFM
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
#%%
A_raw_data=pd.read_csv('/Files/predict poverty/A_raw_data.csv')
B_raw_data=pd.read_csv('/Files/predict poverty/B_raw_data.csv')
C_raw_data=pd.read_csv('/Files/predict poverty/C_raw_data.csv')

B_raw_data_all=pd.read_csv('/Files/predict poverty/B_raw_data_all.csv')
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
        
new_data=data_pool
for i in missing_index:
    count_missing=(data_pool[i].isnull().sum()/data_pool[i].shape[0]).sort_values(ascending=False)
    new_data[i]=new_data[i].drop(list(count_missing.index[count_missing>0.5]),axis=1)

#%%
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
#x' = x/ max(fabs(x))
def bp(data,target,column):
    plt.boxplot([data[column][np.where(target==0)[0]],data[column][np.where(target==1)[0]]],positions=[0,1])
    
def discrepency(data,target,column,index):
    t_1=data[:index].loc[(target==1).values,column].sum()
    t_0=data[:index].loc[(target==0).values,column].sum()
    diff=math.fabs(t_1-t_0)
    count=data[:index].loc[:,column].sum()
    return diff,count
    
def Pearson_select(data,target):
    return SelectKBest(lambda X,Y: np.array(map(lambda x:pearsonr(x,Y),X.T)).T,k=2).fit_transform(data,target)

def normalize(data,column_names):
    for i in column_names:
        data[i]=(data[i]-data[i].mean())/data[i].std()
    return data

def standardize(data,column_names):
    for i in column_names:
        data[i]=(data[i]-data[i].min())/(data[i].max()-data[i].min())
    return data
    
def feature_selection(data,target,k_val):
    test = SelectKBest(score_func=f_classif, k=k_val)
    Y=target
#   X=data  
    X=data
    fit = test.fit(X, Y)
# summarize scores
    np.set_printoptions(precision=3)
    features = fit.transform(X)
    return fit.scores_,features

def LassoLR(data,target,train_index,params):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.2,stratify=target)
#    lrcv=LogisticRegressionCV(Cs=[0.01,0.1,0.5,1,5,10,20],fit_intercept=True,cv=10,penalty='l2',scoring=None, tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv=LogisticRegressionCV(Cs=params,fit_intercept=True,cv=10,penalty='l1',scoring=None,solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv.fit(X_train,y_train)
    y_pred_train=lrcv.predict_proba(X_test)
    y_pred_label=lrcv.predict(X_test)
    acc=np.mean(y_pred_label==y_test)
    y_pred_self=lrcv.predict_proba(X_train)
    train_ll=log_loss(y_train,y_pred_self)
    test_ll=log_loss(y_test,y_pred_train)
    
    clf=LogisticRegression(C=lrcv.C_[0],fit_intercept=True,penalty='l1',solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,multi_class='ovr',random_state=None)
    clf.fit(data.iloc[:train_index,:],target)
    y_pred_test=clf.predict_proba(data.iloc[train_index:,:])
    return train_ll,test_ll,lrcv.C_,y_pred_test,acc,y_pred_train

def RidgeLR(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.15)
#    lrcv=LogisticRegressionCV(Cs=[0.01,0.1,0.5,1,5,10,20],fit_intercept=True,cv=10,penalty='l2',scoring=None, tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv=LogisticRegressionCV(Cs=[0.01,0.1,0.4,0.8],fit_intercept=True,cv=10,penalty='l2',scoring=None, tol=0.0001,max_iter=200,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
    lrcv.fit(X_train,y_train)
    y_pred_train=lrcv.predict_proba(X_test)
    y_pred_label=lrcv.predict(X_test)
    acc=np.mean(y_pred_label==y_test)
    ll=log_loss(y_test,y_pred_train)
    y_pred_test=lrcv.predict_proba(data.iloc[train_index:,:])
    return ll,lrcv.C_,y_pred_test,acc,y_pred_train

def RF(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.2)
    rfc=RandomForestClassifier(criterion='gini',max_features=100,max_depth=10,n_estimators=800)
    rfc.fit(X_train,y_train)
    y_pred=rfc.predict_proba(X_test)
    y_pred_train=rfc.predict_proba(X_train)
    test_ll=log_loss(y_test,y_pred)
    train_ll=log_loss(y_train,y_pred_train)
    return test_ll,train_ll,rfc.feature_importances_
#rfc.feature_importances_

def svm_linear(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    svc=svm.SVC(probability=True)
    parameters = {'kernel':['linear'], 'C':[0.005,0.01,0.05,0.1,1]}
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train,y_train)
    y_pred=clf.predict_proba(X_test)
    ll=log_loss(y_test,y_pred)
    return ll

def svm_rbf(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    svc=svm.SVC(probability=True)
    parameters = {'kernel':['rbf'], 'C':[0.1]}
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train,y_train)
    y_pred=clf.predict_proba(X_test)
    ll=log_loss(y_test,y_pred)
    return ll

def LDA(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    clf = LinearDiscriminantAnalysis(shrinkage='auto')
    clf.fit(X_train, y_train)
    y_pred=clf.predict_proba(X_test)
    ll=log_loss(y_test,y_pred)
    return ll

def NN(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
    model=Sequential()
    model.add(Dense(500,input_shape=(data.shape[1],)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',metrics=["accuracy"], optimizer='adam')
    y_train=np_utils.to_categorical(y_train,2)
    y_test_trans=y_test.copy()
    y_test_trans=np_utils.to_categorical(y_test_trans,2)
    model.fit(X_train, y_train,
          batch_size=100, epochs=8,
          verbose=1,
          validation_data=(X_test, y_test_trans))
    score = model.evaluate(X_test, y_test_trans,verbose=0)
    y_pred=model.predict_proba(X_test, batch_size=50)
    ll=log_loss(y_test_trans,y_pred)
    print('log loss:',ll)    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
def to_norm(data,to_norm_idx):
    temp_df=data[to_norm_idx]
    temp_df_2=data.drop(to_norm_idx,axis=1)   
    temp_df=normalize(temp_df,temp_df.columns)
    new_data=pd.concat([temp_df_2,temp_df],axis=1)
    return new_data

def xgb(data,target,train_index):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:train_index,:], target, test_size=0.25)
#    parameters = {'max_depth':(2,4,6,8,10), 'n_estimators':[400, 600, 800],'reg_alpha'=[0.2,0.4,0.6,0.8]}
    xgb=XGBClassifier(missing=np.nan,min_child_weight=3,learning_rate=0.05,max_depth=8,objective='binary:logistic', n_estimators=200,reg_alpha=0.8)
#    clf = GridSearchCV(svc, parameters,refit=True,cv=5)
    xgb.fit(X_train, y_train)
#    clf.fit(X_train, y_train)
# make predictions for test data
    y_pred_label = xgb.predict(X_test)
    y_pred_prob = xgb.predict_proba(X_test)
    y_pred_prob_train=xgb.predict_proba(X_train)
    train_ll=log_loss(y_train,y_pred_prob_train)
    test_ll=log_loss(y_test,y_pred_prob)
    acc=np.mean(y_pred_label==y_test)
    
    xgb_test=XGBClassifier(min_child_weight=3,learning_rate=0.05,max_depth=8,objective='binary:logistic', n_estimators=800,reg_alpha=0.6)
    xgb_test.fit(data.iloc[:train_index,:],target)
    y_pred_test = xgb_test.predict_proba(data.iloc[train_index:,:])
# evaluate predictions
    return train_ll,test_ll,acc,y_pred_test


def IF(data,target,train_index):  
    IF=IsolationForest(n_estimators=500,max_features=1,max_samples='auto')
    IF.fit(data.loc[:A_train_index,:],target)
    y_pred_outliers = IF.predict(data.loc[:A_train_index,:])
    Z = IF.decision_function(data.loc[:A_train_index,:])
    return y_pred_outliers,Z
#%%  
    
A_train_index=new_data[2].shape[0]
B_train_index=new_data[6].shape[0]
C_train_index=new_data[10].shape[0]

A_id=A_raw_data.id.iloc[A_train_index:]
A_raw_data=A_raw_data.drop('id',axis=1)
A_raw_data=A_raw_data.drop(['family_member','country_A'],axis=1)
A_raw_data=A_raw_data.drop('Unnamed: 0',axis=1)

B_id=B_raw_data.id.iloc[B_train_index:]
B_raw_data=B_raw_data.drop('id',axis=1)
B_raw_data=B_raw_data.drop(['family_member','country_B'],axis=1)
B_raw_data=B_raw_data.drop('Unnamed: 0',axis=1)

C_id=C_raw_data.id.iloc[C_train_index:]
C_raw_data=C_raw_data.drop('id',axis=1)
C_raw_data=C_raw_data.drop(['family_member','country_C'],axis=1)
C_raw_data=C_raw_data.drop('Unnamed: 0',axis=1)
   
A_num_idx=['nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh']
# YFMZwKrU extremely important
# nEsgxvAq very important
# TiwRslOh commonly important
# nEsgxvAq unimportant
B_num_idx=['wJthinfa', 'ZvEApWrk', 'vuQrLzvK', 'qrOrXLPM', 'BXOWgPgL','McFBIGsm', 'NjDdhqIe', 'rCVqiShm', 'ldnyeZwD', 'BEyCyEUG', 'NBWkerdL','BRzuVmyf', 'VyHofjLM', 'GrLBZowF', 'oszSdLhD', 'cDhZjxaW', 'OSmfjCbE','IOMvIGQS']
C_num_idx=['LhUIIEHQ', 'PNAiwXUz', 'jmsRIiqp', 'NONtAKOM','kLAQgdly', 'WWuPOkor', 'CtFxPQPT', 'GIwNbAsH', 'qLDzvjiU', 'detlNNFh','izNLFWMH', 'tXjyOtiS', 'EQtGHLFz', 'xFKmUXhu', 'cmjTMVrd', 'hJrMTBVd','BBPluVrb', 'IRMacrkM', 'EQSmcscG', 'DBjxSUvf', 'kiAJBGqv', 'aFKPYcDt','gAZloxqF', 'phbxKGlB', 'nTaJkLaJ', 'ZZGQNLOX', 'snkiwkvf', 'POJXrpmn','vSqQCatY', 'mmoCpqWS']

A_target=new_data[2].poor
B_target=new_data[6].poor
C_target=new_data[10].poor

A_raw_data=A_raw_data.fillna(method='bfill')

A_raw_data=A_raw_data.drop('country',axis=1)
B_raw_data=B_raw_data.drop('country',axis=1)
C_raw_data=C_raw_data.drop('country',axis=1)

A_idx=list(A_raw_data.dtypes[A_raw_data.dtypes=='object'].index)
#OdXpbPGJ    object
#ukWqmeSS    object
B_idx=list(B_raw_data.dtypes[B_raw_data.dtypes=='object'].index)
#TJGiunYp      object
#esHWAAyG      object
#gKsBCLMY      object
#TZDgOhYY      object
#jzBRbsEG      object
#dnmwvCng      object
#wJthinfa_y    object
#mAeaImix      object
#ulQCDoYe      object
C_idx=list(C_raw_data.dtypes[C_raw_data.dtypes=='object'].index)
#XKQWlRjk    object  
#vWNISgEA    object  
#bsMfXBld    object
#XKyOwsRR    object
#CgAkQtOd    object all take average

#C_all_idx=((C_raw_data[C_num_idx[0]]>0).all()) or ((C_raw_data[C_num_idx[0]]<0).all())
#B_all_idx=((B_raw_data[B_num_idx[0]]>0).all()) or ((B_raw_data[B_num_idx[0]]<0).all())
#A_all_idx=((A_raw_data[A_num_idx[0]]>0).all()) or ((A_raw_data[A_num_idx[0]]<0).all())


p1=re.compile(r'\d+\.\d+|nan')
p2=re.compile(r'\d+')
p=re.compile(r'\d+\.\d+|\d+|nan')

def nan_float(data_column,replacement):
    for i in data_column:
        for j in range(len(i)):
            if i[j] == 'nan':
                i[j]=replacement
            i[j]=float(i[j])
    return data_column

def nan_int(data_column):
    for i in data_column:
        for j in range(len(i)):
            i[j]=int(i[j])
    return data_column


#%%
#fix A missing values and calculation

A_raw_data.OdXpbPGJ=A_raw_data.OdXpbPGJ.apply(lambda x: p1.findall(x))
A_raw_data.ukWqmeSS=A_raw_data.ukWqmeSS.apply(lambda x: p2.findall(x))
A_raw_data.OdXpbPGJ=nan_float(A_raw_data.OdXpbPGJ,4.0)
A_raw_data.ukWqmeSS=nan_int(A_raw_data.ukWqmeSS)
A_raw_data.OdXpbPGJ=A_raw_data.OdXpbPGJ.apply(lambda x:np.mean(np.array(x)))
A_raw_data.ukWqmeSS=A_raw_data.ukWqmeSS.apply(lambda x:np.mean(np.array(x)))
A_raw_data=A_raw_data.drop('Unnamed: 0',axis=1)
A_raw_data=A_raw_data.drop(['country','country_A','id'],axis=1)
A_raw_data=A_raw_data.drop(['family_member'],axis=1)
A_todrop_idx=['OMtioXZZ','ukWqmeSS']
#['WKBXv','XJgvq','jTlga','pkNwY']
#%%
pca=PCA(n_components=1000)
new_df=ca.fit_transform(A_try)
A_test_df=pd.DataFrame(new_df)
pca.explained_variance_ratio_.cumsum()
#%%
test['new1']=(-test[test.columns[2]]+test[test.columns[0]])
plt.boxplot([test[A_target==0]['new1'],test[A_target==1]['new1']],positions=[0,1])

#%%
df=pd.concat([new_data[4],new_data[3]],axis=0)
iter_idx=list(df.columns)
iter_idx.remove('OdXpbPGJ')
iter_idx.remove('ukWqmeSS')
iter_idx.remove('country')
iter_idx.remove('id')
iter_idx.remove('iid')
iter_idx.remove('poor')
temp_dict={}
for k in iter_idx:
    temp_list=[]
    for i in df.id.unique():
        temp=list(df[df.columns[2]][df.id==i])
        temp=Counter(temp)
        temp=sorted(temp.items(),key=lambda x:x[1])
        temp_list.append(temp[0][0])        
    temp_dict[k]=temp_list
new_add_df=pd.DataFrame(temp_dict)
A_try=A_raw_data.copy()
new_add_df=pd.get_dummies(new_add_df)
A_try=pd.concat([A_try,new_add_df],axis=1)
#%%
order_idx={}
for i in A_raw_data.columns:
    tau,p_val=kendalltau(A_raw_data.loc[:A_train_index-1,i].values,A_target.values)
    order_idx[i]=math.fabs(tau)
sorted_order_idx=sorted(order_idx.items(),key=lambda x:x[1])
A_kandall_idx=[]
for i in sorted_order_idx:
    if i[-1]<0.01:
        A_kandall_idx.append(i[0])
#%%
#A_try=A_raw_data.copy()
df_test=A_raw_data[A_raw_data.columns[846:]]
convert_feature=list(df_test.columns)
#convert_feature.extend(['nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh'])
#convert_data=A_try[convert_feature]
convert_data=A_try
X_train, X_test, y_train, y_test = train_test_split(convert_data.iloc[:A_train_index,:], A_target, test_size=0.2)
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 80,
	'num_trees': 1000,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,lgb_train,num_boost_round=300,valid_sets=lgb_train)

# save model to file
#gbm.save_model('model.txt')
#y_pred = gbm.predict(X_train,pred_leaf=True)
y_pred_prob= gbm.predict(X_test,pred_leaf=True)
y_pred_prob1= gbm.predict(X_test)
y_pred_prob0=1-y_pred_prob1
a=y_pred_prob1.reshape((len(y_pred_prob1),1))
b=y_pred_prob0.reshape((len(y_pred_prob0),1))
y_prob=np.concatenate((b,a),axis=1)
#y_pred_prob_0=1-y_pred_prob_1
y_pred=[]
for i in y_pred_prob1:
    if i>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(np.mean(y_pred==y_test))
print(log_loss(y_test,y_prob))
#%%
A_convert_data=A_data[df_test.columns]
X_train, X_test, y_train, y_test = train_test_split(A_convert_data.iloc[:A_train_index,:], A_target, test_size=0.2)
A_convert_data=A_data[df_test.columns]
X_train, X_test, y_train, y_test = train_test_split(A_convert_data.iloc[:A_train_index,:], A_target, test_size=0.2)
gbc=GradientBoostingClassifier(n_estimators=200,min_samples_leaf=70,max_features='sqrt',loss='deviance')
param_test = {'max_depth':[i for i in range(5,16,2)], 'min_samples_split':[i for i in range(200,1001,200)]}
gsearch = GridSearchCV(estimator = gbc, param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
ohe=OneHotEncoder()
gsearch.fit(X_train,y_train)
print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
print(np.mean(gsearch.predict(X_test)==y_test))

#%%
#A_numerical_idx=list(A_raw_data[A_raw_data.columns[846:]].columns)
#A_cate_idx=list(A_raw_data[A_raw_data.columns[:846]].columns)
A_numerical_idx=['nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh','OMtioXZZ','ukWqmeSS']
A_cate_idx=[i for i in A_raw_data.columns if i not in A_numerical_idx]

#A_rank=A_try.apply(lambda x: (x==0).sum()/A_try.shape[0]).sort_values(ascending=False)

A_try=A_raw_data.copy()
#A_try=A_try.drop(A_rank.index[:40],axis=1)
#
#for i in A_rank.index[:40]:
#    try:
#        A_numerical_idx.remove(i)
#    except:
#        continue
#
#for i in A_rank.index[:40]:
#    try:
#        A_cate_idx.remove(i)
#    except:
#        continue
                
A_numerical=A_try[A_numerical_idx]
A_cate=A_try[A_cate_idx]

X_train_tree, X_train_lr, y_train_tree, y_train_lr = train_test_split(A_try.iloc[:A_train_index,:], A_target, test_size=0.75)
X_train, X_test, y_train, y_test = train_test_split(X_train_lr.iloc[:A_train_index,:], y_train_lr, test_size=0.2)

gbc=GradientBoostingClassifier(n_estimators=150,max_depth=6,min_samples_leaf=50,max_features='sqrt',loss='deviance',min_samples_split=400)
ohe=OneHotEncoder()
gbc.fit(X_train_tree[A_numerical_idx],y_train_tree)
#print(np.mean(gbc.predict(X_test)==y_test))
ohe.fit(gbc.apply(X_train_tree[A_numerical_idx])[:,:,0])

convert_data_train=ohe.transform(gbc.apply(X_train[A_numerical_idx])[:,:,0])
convert_data_train=pd.DataFrame(convert_data_train.toarray())
convert_data_train=pd.concat([X_train.reset_index(drop=True),convert_data_train.reset_index(drop=True)],axis=1)
#y_train
convert_data_test=ohe.transform(gbc.apply(X_test[A_numerical_idx])[:,:,0])
convert_data_test=pd.DataFrame(convert_data_test.toarray())
convert_data_test=pd.concat([X_test.reset_index(drop=True),convert_data_test.reset_index(drop=True)],axis=1)

convert_data_test=standardize(convert_data_test,A_numerical_idx)
convert_data_train=standardize(convert_data_train,A_numerical_idx)

params=[0.0001,0.001,0.01,0.1,0.2,0.4,0.6,0.8]

#convert_data_train=standardize(convert_data_train,convert_data_train.columns)
#convert_data_train=convert_data_train.dropna(axis=1,how='all')
#convert_data_test=standardize(convert_data_test,convert_data_test.columns)
#convert_data_test=convert_data_test.dropna(axis=1,how='all')

lrcv=LogisticRegressionCV(Cs=params,fit_intercept=True,cv=10,penalty='l1',scoring=None,solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
lrcv.fit(convert_data_train,y_train)
y_pred_train=lrcv.predict_proba(convert_data_train)
y_pred_test=lrcv.predict_proba(convert_data_test)

y_pred_label=lrcv.predict(convert_data_test)
acc=np.mean(y_pred_label==y_test)

train_ll=log_loss(y_train,y_pred_train)
test_ll=log_loss(y_test,y_pred_test)
print('train log loss: ',train_ll)
print('test log loss: ',test_ll)
print('accuracy: ',acc)
#%%
clf=LogisticRegression(C=lrcv.C_[0],fit_intercept=True,penalty='l1',solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,multi_class='ovr',random_state=None)
convert_data_all=ohe.transform(gbc.apply(A_try.loc[:A_train_index-1,A_numerical_idx])[:,:,0])
convert_data_all=pd.DataFrame(convert_data_all.toarray())
convert_data_all=pd.concat([A_try.loc[:A_train_index-1,A_cate_idx].reset_index(drop=True),convert_data_all.reset_index(drop=True)],axis=1)

convert_data_target=ohe.transform(gbc.apply(A_try.loc[A_train_index:,A_numerical_idx])[:,:,0])
convert_data_target=pd.DataFrame(convert_data_target.toarray())
convert_data_target=pd.concat([A_try.loc[A_train_index:,A_cate_idx].reset_index(drop=True),convert_data_target.reset_index(drop=True)],axis=1)

A_all_data=pd.concat([convert_data_all,convert_data_target],axis=0)
A_train_ll,A_test_ll,A_c,A_result,A_acc,_=LassoLR(A_all_data,A_target,A_train_index,[0.001,0.01,0.1,0.2,0.4])
#clf.fit(convert_data_all,A_target)
#A_result=clf.predict_proba(convert_data_target)

#%%
#df_test.columns[19]
#'CNSXC'
#'olfwp'
#df_test.columns[213] >0 is 1 or is 0
#df_test.columns[240] >0 is 1 or is 0
#df_test.columns[246] >0 is 1 or is 0
#df_test.columns[77] >0 is 1 or is 0
#df_test.columns[81] >0 is 1 or is 0
#df_test.columns[90] >0 is 1 or is 0
#df_test.columns[97] >0 is 1 or is 0
#df_test.columns[68] >0 is 1 or is 0
#df_test.columns[51] >0 is 1 or is 0
#df_test.columns[47] >0 is 1 or is 0
#df_test.columns[52] >0 is 1 or is 0
#df_test.columns[58] >0.5 is 1 or is 0
#df_test.columns[61] >0.5 is 1 or is 0
#df_test.columns[67] >0.4 is 1 or is 0
#df_test.columns[71] >0.8 is 1 or is 0
#df_test.columns[44] >0.8 is 1 or is 0
#df_test.columns[46] >0.8 is 1 or is 0
#%%
best_idx=[2,4,8,17,33,52,82,113,124,125,127,169,210,215,240,247]
#best_idx=[2,3,4,5,6,8,17,23,24,25,27,33,34,37,38,39,42,51,52,63,76,82,84,85,95,102,112,113,114,120,124,125,127,133,143,150,151,157,165,169,170,176,184,191,198,206,208,210,215,226,227,228,234,237,240,243,244,247,253,254,260,268,269,272,274]
best_feature=[]
for i in best_idx:
    best_feature.append(df_test.columns[i])
#%%
df_test=A_raw_data[A_raw_data.columns[846:]]
A_need_drop=[0,1,10,11,22,32,46,57,58,59,67,69,70,77,78,80,83,86,91,92,97,98,108,109,110,111,115,119,121,132,134,148,149,155,159,164,166,177,187,188,190,196,200,201,219,229,230,231,233,252,253,263,266,271,273]
#A_need_drop=[0,1,10,11,19,20,22,29,32,35,43,45,46,55,57,58,59,63,67,69,70,77,78,80,83,86,91,92,97,98,108,109,110,111,115,119,121,132,134,148,149,155,159,160,164,166,177,187,188,190,196,200,201,219,229,230,231,233,252,253,263,266,271,273]
A_needrop_idx=[]
for i in A_need_drop:
    A_needrop_idx.append(df_test.columns[i])
A_left=[i for i in range(len(df_test.columns)) if i not in A_need_drop]
A_tonorm_idx=[]
for i in A_left:
    A_tonorm_idx.append(df_test.columns[i])
A_tonorm_idx.remove('OMtioXZZ')
A_tonorm_idx.remove('country')
A_tonorm_idx.extend(['nEsgxvAq', 'YFMZwKrU', 'TiwRslOh'])


A_try=A_raw_data.copy()
A_try=A_raw_data.drop(A_needrop_idx,axis=1)
A_try=A_try.drop('country',axis=1)
A_try=A_try.drop(['OMtioXZZ'],axis=1)
#%%
A_check_idx=list(A_raw_data.columns[1:847])
discrep={}
for i in A_check_idx: 
    diff,count=discrepency(A_raw_data,A_target,i,A_train_index)
    discrep[i]=[diff,count,diff/count]
sorted_discrep=sorted(discrep.items(),key=lambda x:x[1][-1])
A_drop_hhold_idx=[]
for i in range(len(sorted_discrep)):
    if sorted_discrep[i][-1][-1]<0.01:
    #or sorted_discrep[i][-1][-1]>0.995:
        A_drop_hhold_idx.append(sorted_discrep[i][0])
#A_hhold_part=A_keep_origin[A_keep_origin.columns[:846]]  
#%%
df_test_hhold=A_raw_data[A_raw_data.columns[:847]]
num_list=[]
for i in df_test_hhold.columns:
    num=(df_test_hhold[i]==1).sum()
    num_list.append(num)
idx=np.where(np.array(num_list)<10)[0]
A_bf_idx=[]
for i in idx[:-1]:
    A_bf_idx.append(df_test_hhold.columns[i])


df_test=A_raw_data[A_raw_data.columns[847:]]
#A_try=A_raw_data.copy()
A_impo=[2,3,4,5,6,8,17,23,24,25,27,33,34,37,38,39,42,51,52,63,76,82,84,85,95,102,112,113,114,120,124,125,127,133,143,150,151,157,165,169,170,176,184,191,198,206,208,210,215,226,227,228,234,237,240,243,244,247,253,254,260,268,269,272,274]
A_unimpo=[i for i in range(len(df_test.columns)) if i not in A_impo]
A_tonorm_idx1=[]
for i in A_impo:
    A_tonorm_idx1.append(df_test.columns[i])
A_tonorm_idx1.remove('family_member')
A_tonorm_idx1.remove('OMtioXZZ')
A_tonorm_idx1.extend(['nEsgxvAq', 'YFMZwKrU', 'TiwRslOh'])
A_unimpo_idx=[]
for i in A_unimpo:
    A_unimpo_idx.append(df_test.columns[i])

A_try=A_raw_data.drop(A_unimpo_idx,axis=1)
A_try=A_try.drop(['OMtioXZZ'],axis=1)
A_try=A_try.drop('Unnamed: 0',axis=1)
A_raw_data=A_raw_data.drop(['family_member'],axis=1)
A_raw_data=A_raw_data.drop(['country','country_A','id'],axis=1)
#A_try=A_try.drop(A_drop_hhold_idx,axis=1)
#A_impo[47] 'olfwp'   delete >0.8
#A_impo[29] 'VzUws'   delete =1.0
#A_impo[0] 'TiwRslOh' delete <-20 (or <-15)
### 'YFMZwKrU' delete >=0
# 'nEsgxvAq' delete <=60.0
A_tt=A_raw_data.copy()
A_tt=A_tt.drop(A_raw_data.loc[:A_train_index,:].index[(A_raw_data.loc[:A_train_index,'TiwRslOh']<=-20).values],axis=0)
A_tt=A_tt.drop(A_raw_data.loc[:A_train_index,:].index[(A_raw_data.loc[:A_train_index,'olfwp']>0.8).values],axis=0)
A_tt=A_tt.drop(A_raw_data.loc[:A_train_index,:].index[(A_raw_data.loc[:A_train_index,'VzUws']==1).values],axis=0)
#A_tt=A_tt.drop(A_raw_data.loc[:A_train_index,:].index[(A_raw_data.loc[:A_train_index,'nEsgxvAq']<=60.0).values],axis=0)
#%%
A_tonorm_idx=[i for i in A_num_idx if i not in A_todrop_idx]
A_tonorm_idx.append('OdXpbPGJ')

df_test_hhold=df_test_hhold.drop('Unnamed: 0',axis=1)
hhold_cols=df_test_hhold.columns
A_dropimpo_1=[]
for i in hhold_cols:
    t_1=df_test_hhold[:A_train_index].loc[(A_target==1).values,i].sum()
    t_0=df_test_hhold[:A_train_index].loc[(A_target==0).values,i].sum()
    diff=math.fabs(t_1-t_0)
    if diff<100.0:
        A_dropimpo_1.append(i)
A_new_hhold=df_test_hhold.drop(A_dropimpo_1,axis=1)

cols=df_test.dtypes[df_test.dtypes=='object'].index
#df_check=pd.get_dummies(df_test[cols])
df_check=df_test.drop(['country','country_A','id'],axis=1)
#df_check['id']=df_test.id.values
#df_check_copy=df_check.groupby('id').sum()

A_dropimpo=[]
for i in df_check.columns:
    t_1=df_check[:A_train_index].loc[(A_target==1).values,i].sum()
    t_0=df_check[:A_train_index].loc[(A_target==0).values,i].sum()
    diff=math.fabs(t_1-t_0)
    if diff<300.0:
        A_dropimpo.append(i)
    
A_new_indiv=df_check.drop(A_dropimpo,axis=1)
    
A_try=pd.concat([A_new_indiv,A_new_hhold],axis=1)

A_cormat=A_try.corr().abs()

#cols=A_try.corr().abs().columns.values
#for col, row in (A_cormat > 0.7).iteritems():
#    print(col, cols[row.values])

#A_try.corr().abs().columns.values
#A_cormat=A_cormat.unstack()
check_hc=A_cormat.sort_values(ascending=False)
#for i in check_hc:
#    print(i)
columns=[]
for i in range(len(check_hc>0.7)):
    columns.append(list(check_hc.index[i]))
df_hc_idx=pd.DataFrame(np.array(columns))

#A_try=A_try.drop(A_todrop_idx,axis=1)
#%%
#fix B missing values and calculation
flexible_idx=['BRzuVmyf','jzBRbsEG','TJGiunYp','esHWAAyG','TZDgOhYY']
still_missing=['McFBIGsm','BXOWgPgL','OSmfjCbE']
B_try=B_raw_data.drop(flexible_idx,axis=1)
B_try[still_missing]=B_try[still_missing].fillna(B_try[still_missing].mean())

B_try=B_raw_data.copy()
missing_idx=['BRzuVmyf','jzBRbsEG','TJGiunYp','esHWAAyG','TZDgOhYY','McFBIGsm','BXOWgPgL','OSmfjCbE']
B_try[missing_idx]=B_try[missing_idx].fillna(0)
#%%
B_try=B_raw_data[0].copy()
missing_idx=['BRzuVmyf','jzBRbsEG','TJGiunYp','esHWAAyG','TZDgOhYY','McFBIGsm','BXOWgPgL','OSmfjCbE']
B_try=B_try.fillna(0)

for i in B_try.columns:
    try:
       B_try[i]=B_try[i].apply(lambda x: p.findall(x))
    except:
        print('no need') 

#new_B_idx=list(B_try.dtypes[B_try.dtypes=='object'].index)
#for i in new_B_idx:    
#    B_try[i]=B_try[i].apply(lambda x: p.findall(x))
    
for i in B_try.columns:
    try:
        B_try[i]=nan_float(B_try[i],0)
    except:
        print('no need')

for i in B_try.columns:
    try:
        for j in range(len(B_try[i])):
            temp=B_try[i][j].copy()
            while np.nan in temp:
                temp.remove(np.nan)
            if temp == []:
                fill_val=0
            else:
                fill_val=np.mean(np.array(temp))            
            new_list=[fill_val if x is np.nan else x for x in B_try[i][j]]
            B_try.loc[j,i]=np.mean(np.array(new_list))
    except:
        print('no need')
for i in B_try.columns:
    try:
        B_try[i]= B_try[i].apply(lambda x:float(x)) 
    except:
        print(i)
B_try.jzBRbsEG=B_try.jzBRbsEG.apply(lambda x:np.array(x).mean())

B_try=B_try.drop('country',axis=1)
B_try=B_try.drop('Unnamed: 0',axis=1)
B_try=B_try.drop('id',axis=1)
B_try=B_try.fillna(0)

#%%
#A_numerical_idx=A_raw_data[A_raw_data.columns[846:]].columns
#A_cate_idx=A_raw_data[A_raw_data.columns[:846]].columns
#to_add_idx=['nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh']
#A_numerical=A_try[A_numerical_idx]
#A_cate=A_try[A_cate_idx]
X_train_tree, X_train_lr, y_train_tree, y_train_lr = train_test_split(B_try.iloc[:B_train_index,:], B_target, test_size=0.6,stratify=B_target)
X_train, X_test, y_train, y_test = train_test_split(X_train_lr.iloc[:B_train_index,:], y_train_lr, test_size=0.2,stratify=y_train_lr)

gbc=GradientBoostingClassifier(n_estimators=50,max_depth=6,min_samples_leaf=100,max_features='sqrt',loss='deviance',min_samples_split=800)
ohe=OneHotEncoder()
gbc.fit(X_train_tree,y_train_tree)
#print(np.mean(gbc.predict(X_test)==y_test))
ohe.fit(gbc.apply(X_train_tree)[:,:,0])

convert_data_train=ohe.transform(gbc.apply(X_train)[:,:,0])
convert_data_train=pd.DataFrame(convert_data_train.toarray())
convert_data_train=pd.concat([X_train.reset_index(drop=True),convert_data_train.reset_index(drop=True)],axis=1)
#y_train
convert_data_test=ohe.transform(gbc.apply(X_test)[:,:,0])
convert_data_test=pd.DataFrame(convert_data_test.toarray())
convert_data_test=pd.concat([X_test.reset_index(drop=True),convert_data_test.reset_index(drop=True)],axis=1)

params=[0.000001,0.0001,0.001,0.01,0.1,0.2,0.4,0.6,0.8]

#convert_data_train=standardize(convert_data_train,convert_data_train.columns)
#convert_data_train=convert_data_train.dropna(axis=1,how='all')
#convert_data_test=standardize(convert_data_test,convert_data_test.columns)
#convert_data_test=convert_data_test.dropna(axis=1,how='all')

lrcv=LogisticRegressionCV(Cs=params,fit_intercept=True,cv=10,penalty='l1',scoring=None,solver='liblinear', tol=0.0001,max_iter=100,class_weight=None,n_jobs=1,verbose=0,refit=True,multi_class='ovr',random_state=None)
lrcv.fit(convert_data_train,y_train)
y_pred_train=lrcv.predict_proba(convert_data_train)
y_pred_test=lrcv.predict_proba(convert_data_test)

y_pred_label=lrcv.predict(convert_data_test)
acc=np.mean(y_pred_label==y_test)

train_ll=log_loss(y_train,y_pred_train)
test_ll=log_loss(y_test,y_pred_test)
print('train log loss: ',train_ll)
print('test log loss: ',test_ll)
print('accuracy: ',acc)

#%% 
bug_idx=['gKsBCLMY', 'dnmwvCng', 'wJthinfa_y', 'mAeaImix', 'ulQCDoYe']
for i in bug_idx:
    B_data[i]= B_data[i].apply(lambda x:float(x)) 
#%%
for i in B_try.columns:
    B_try[i]= B_try[i].apply(lambda x:float(x))  
B_try=B_try.drop(['country','country_B'],axis=1)
#%%
#for i in new_B_idx:  
#    B_try[i]=B_try[i].apply(lambda x: np.mean(np.array(x)))   
B_data=B_try.copy()
#B_to_drop=[1,2,3,4,5,6,7,8,9,10,13,14,15,17]
B_to_drop=[1,3,9,10,14,15]
B_todrop_idx=[]
for i in B_to_drop:
    B_todrop_idx.append(B_num_idx[i])
for i in [0,1,2]:
    B_todrop_idx.append(new_B_idx[i])
for i in B_todrop_idx:
    if i not in B_data.columns:
        B_todrop_idx.remove(i)

B_tonorm_idx=[i for i in B_num_idx if i not in B_todrop_idx]
B_tonorm_idx.extend(['mAeaImix','ulQCDoYe'])
for i in B_tonorm_idx:
    if i not in B_data.columns:
        B_tonorm_idx.remove(i)
B_tonorm_idx.remove('BRzuVmyf')
    
B_data=B_data.drop(B_todrop_idx,axis=1)
#%%
#convert_feature.extend(['nEsgxvAq', 'OMtioXZZ', 'YFMZwKrU', 'TiwRslOh'])
#convert_data=A_try[convert_feature]
still_missing=['McFBIGsm','BXOWgPgL','OSmfjCbE']
#B_try=B_data
#B_try[still_missing]=B_try[still_missing].fillna(B_try[still_missing].mean())
fix_idx=['ulQCDoYe']
#'mAeaImix'
B_try['mAeaImix']=B_try['mAeaImix'].apply(lambda x:float(x))
B_try['ulQCDoYe']=B_try['ulQCDoYe'].apply(lambda x:float(x))
X_train, X_test, y_train, y_test = train_test_split(B_try.iloc[:B_train_index,:], B_target, test_size=0.2)
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 35,
	'num_trees': 600,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0
}
params['is_unbalance']='true'
gbm = lgb.train(params,lgb_train,num_boost_round=300,valid_sets=lgb_train)

# save model to file
#gbm.save_model('model.txt')
#y_pred = gbm.predict(X_train,pred_leaf=True)
y_pred_prob= gbm.predict(X_test,pred_leaf=True)
y_pred_prob1= gbm.predict(X_test)
y_pred_prob0=1-y_pred_prob1
a=y_pred_prob1.reshape((len(y_pred_prob1),1))
b=y_pred_prob0.reshape((len(y_pred_prob0),1))
y_prob=np.concatenate((b,a),axis=1)
#y_pred_prob_0=1-y_pred_prob_1
y_pred=[]
for i in y_pred_prob1:
    if i>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(np.mean(y_pred==y_test))
print(log_loss(y_test,y_prob))


#%%
#fix C missing values and calculation
for i in C_idx:
    C_raw_data[i]=C_raw_data[i].apply(lambda x: p.findall(x))
for i in C_idx:
    C_raw_data[i]=nan_float(C_raw_data[i],0)
    C_raw_data[i]=C_raw_data[i].apply(lambda x: np.mean(np.array(x)))

score,features=feature_selection(C_raw_data.loc[:C_train_index-1,C_num_idx],C_target,30)

C_try=C_raw_data.copy()

#test_df=pd.concat([C_try[:C_train_index],C_target],axis=1)
#for i in C_num_idx:
#    print(C_num_idx.index(i),'  ','True:',np.mean(test_df.loc[C_target==1,i]),'False:',np.mean(test_df.loc[C_target==0,i]))

#for i in C_num_idx:  
#    C_try[i+'_01']=C_raw_data[i].values
#    C_try[i+'_01']=C_try[i+'_01'].apply(lambda x:1 if x>0 else 0)
#C_check_idx=[i+'_01' for i in C_num_idx]
#C_to_drop_idx=np.array(C_check_idx)[((C_try[C_check_idx].sum()==C_try.shape[0])|(C_try[C_check_idx].sum()==0)).values]
#C_try=C_try.drop(C_to_drop_idx,axis=1)
#C_scale_idx=[7,13,19,20,27]

C_less_impo=[0,1,3,4,6,9,12,18,24,28,29]#not sure 13,22,26
C_unimpo_idx=[]
for i in C_less_impo:
    C_unimpo_idx.append(C_num_idx[i]) 
    
C_try=C_try.drop(C_unimpo_idx,axis=1)

C_tonorm_idx=[i for i in C_num_idx if i not in C_unimpo_idx]
#C_ohe_idx=['LhUIIEHQ','PNAiwXUz','jmsRIiqp','NONtAKOM','kLAQgdly','CtFxPQPT','qLDzvjiU','detlNNFh','izNLFWMH','hJrMTBVd','EQSmcscG','aFKPYcDt','gAZloxqF','phbxKGlB','nTaJkLaJ','ZZGQNLOX','snkiwkvf','vSqQCatY']
#C_scale_idx=[i for i in C_num_idx if i not in C_ohe_idx and i not in ['EQtGHLFz','cmjTMVrd','mmoCpqWS']]


#for i in C_num_idx:
#    #if (C_try[i]>0).all()==1 and len(set(C_try[i]))<=10:
#    if len(set(C_try[i]))<=100:
#        C_ohe_idx.append(i)
#    else:
#        C_scale_idx.append(i)
    
#C_temp=C_try.drop(C_ohe_idx,axis=1)
#C_temp2=C_try[C_ohe_idx]
#C_temp2=C_temp2.applymap(lambda x: str(x))
#C_try=pd.merge(C_temp,pd.get_dummies(C_temp2),right_index=True,left_index=True)
#%%
C_try=C_data
X_train, X_test, y_train, y_test = train_test_split(C_data.iloc[:C_train_index,:], C_target, test_size=0.2)
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
	'num_trees': 500,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,lgb_train,num_boost_round=300,valid_sets=lgb_train)

# save model to file
#gbm.save_model('model.txt')
#y_pred = gbm.predict(X_train,pred_leaf=True)
y_pred_prob= gbm.predict(X_test,pred_leaf=True)
y_pred_prob1= gbm.predict(X_test)
y_pred_prob0=1-y_pred_prob1
a=y_pred_prob1.reshape((len(y_pred_prob1),1))
b=y_pred_prob0.reshape((len(y_pred_prob0),1))
y_prob=np.concatenate((b,a),axis=1)
#y_pred_prob_0=1-y_pred_prob_1
y_pred=[]
for i in y_pred_prob1:
    if i>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(np.mean(y_pred==y_test))
print(log_loss(y_test,y_prob))
#%%
#Testing Algorithm
#add_idx=['nEsgxvAq','OMtioXZZ','YFMZwKrU','TiwRslOh']
#drop_idx=['nEsgxvAq_0.0', 'nEsgxvAq_0.1875', 'nEsgxvAq_0.25', 'nEsgxvAq_0.3125',
#       'nEsgxvAq_0.375', 'nEsgxvAq_0.4375', 'nEsgxvAq_0.5', 'nEsgxvAq_0.5625',
#       'nEsgxvAq_0.625', 'nEsgxvAq_0.6875', 'nEsgxvAq_0.75', 'nEsgxvAq_0.8125',
#       'nEsgxvAq_0.875', 'nEsgxvAq_0.9375', 'nEsgxvAq_1.0', 'OMtioXZZ_0.0',
#       'OMtioXZZ_0.5084033613445378', 'OMtioXZZ_0.5462184873949579',
#       'OMtioXZZ_0.5840336134453782', 'OMtioXZZ_0.6218487394957983',
#       'OMtioXZZ_0.6596638655462185', 'OMtioXZZ_0.6974789915966386',
#       'OMtioXZZ_0.7352941176470589', 'OMtioXZZ_0.773109243697479',
#       'OMtioXZZ_0.8109243697478992', 'OMtioXZZ_0.8487394957983193',
#       'OMtioXZZ_0.9243697478991597', 'OMtioXZZ_1.0', 'YFMZwKrU_0.0',
#       'YFMZwKrU_0.2', 'YFMZwKrU_0.4', 'YFMZwKrU_0.6', 'YFMZwKrU_0.8',
#       'YFMZwKrU_1.0', 'TiwRslOh_0.0', 'TiwRslOh_0.05555555555555555',
#       'TiwRslOh_0.1111111111111111', 'TiwRslOh_0.16666666666666666',
#       'TiwRslOh_0.2222222222222222', 'TiwRslOh_0.2777777777777778',
#       'TiwRslOh_0.3333333333333333', 'TiwRslOh_0.3888888888888889',
#       'TiwRslOh_0.4444444444444444', 'TiwRslOh_0.5',
#       'TiwRslOh_0.5555555555555556', 'TiwRslOh_0.6111111111111112',
#       'TiwRslOh_0.6666666666666666', 'TiwRslOh_0.7222222222222222',
#       'TiwRslOh_0.7777777777777778', 'TiwRslOh_0.8333333333333334',
#       'TiwRslOh_0.8888888888888888', 'TiwRslOh_0.9444444444444444',
#       'TiwRslOh_1.0']
#new_add=pd.concat([new_data[2][add_idx],new_data[1][add_idx]],axis=0)
#A_raw_data_temp=A_raw_data.drop(drop_idx,axis=1)
#A_raw_data=pd.concat([A_raw_data_temp,new_add],axis=1)
#A_data=standardize(A_try,A_try.columns)
#%%
A_data=to_norm(A_try,A_tonorm_idx)
A_var_diff=A_data.apply(lambda x:np.var(x)).sort_values()
#A_diff=A_data.apply(lambda x:len(x[x!=0])).sort_values()
A_drop_idx2=A_var_diff.index[A_var_diff<0.001]

A_data=A_data.drop(A_drop_idx2,axis=1)
#A_score,_=feature_selection(A_data[:A_train_index],A_target,len(A_data.columns))


#A_feature_impo={}
#for i in range(len(A_data.columns)):
#    A_feature_impo[A_data.columns[i]]=A_score[i]
#A_impo_rank=sorted(feature_impo.items(),key=lambda item:item[1],reverse=True)
#
#rank=[]
#for k in [200,400,600,800,1000]:
#    cols=[]
#    for i in range(k):
#        cols.append(A_impo_rank[:k][i][0])
#    test_data=A_data[cols]
#    A_ll,A_c,A_result,A_acc,A_y_pred_train=LassoLR(A_data,A_target,A_train_index,[0.01,0.1,0.2,0.4,0.6])
#    rank.append(A_ll)
#%%
#A_data=standardize(A_try,A_try.columns)
A_data=to_norm(A_try,A_try.columns)
A_train_ll,A_test_ll,A_c,A_result,A_acc,A_y_pred_train=LassoLR(A_data,A_target,A_train_index,[0.1,0.2,0.4,0.6,0.8,1])
A=pd.DataFrame({'poor':A_result[:,-1]})
A['id']=A_id.values
A['country']='A'
A=A.reindex(columns=['id','country','poor'])
A.to_csv('/Files/predict poverty/A_result.csv')
#%%
#%%    
#B_data=standardize(B_data,B_data.columns)
B_data=to_norm(B_data,B_tonorm_idx)
B_data=B_data.drop('country',axis=1)
B_train_ll,B_test_ll,B_c,B_result,B_acc,B_y_pred_train=LassoLR(B_data,B_target,B_train_index,[0.01,0.1,0.2,0.4,0.6])
B=pd.DataFrame({'poor':B_result[:,-1]})
B['id']=B_id.values
B['country']='B'
B=B.reindex(columns=['id','country','poor'])
B.to_csv('/Files/predict poverty/B_result.csv')
#%%
#C_data=standardize(C_try,C_try.columns)
C_data=to_norm(C_try,C_tonorm_idx)
C_ll,C_c,C_result,C_acc,C_y_pred_train=LassoLR(C_data,C_target,C_train_index,[0.6,0.8,1,1.2,1.5])
C=pd.DataFrame({'poor':C_result[:,-1]})
C['id']=C_id.values
C['country']='C'
C=C.reindex(columns=['id','country','poor'])
C.to_csv('/Files/predict poverty/C_result.csv')
#%%
C_train_ll,C_test_ll,C_acc,C_result=xgb(C_data,C_target,C_train_index)


