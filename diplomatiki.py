#!/usr/bin/env python
# coding: utf-8

# In[134]:


import os
import sys
import pandas as pd
import arff
import pandas as pd
import numpy as np
import math
from statistics import *
from pymfe.mfe import MFE
from statistics import *
from scipy.stats import skew
from scipy.stats import kurtosis
from pandas import DataFrame


# In[166]:


def extract_metafeature(a):
    #statistical(3)
    #print(mean(a.kurtosis()))
    #print(mean(a.skew()))
    #print (mean(a.mean()))
    from sklearn.feature_selection import mutual_info_classif
    from info_gain import info_gain
    y=df[2000]
    X = df.drop(2000,1)
    ft2 = pd.DataFrame({
     #simple
    'nr_instances':[len(a)],
    'nr_features':[len(a.columns)],
    'nr_missing_values':[a.isnull().sum().sum()],
    #statistical 
    #"max_value":[a.values.max()],
    #"min_value":[a.values.min()],
    'mean_kurtosis':[mean(a.kurtosis())],
    'mean_skewness':[mean(a.skew())],
    'mean':[mean(a.mean())],
    #information_theoretic
    #'MI':[mean(mutual_info_classif(X, y))],
    #model_based
    'Info_gain':[info_gain.info_gain(X,y)],
    #'Intistic_value':[info_gain.intrinsic_value(X,y)],
    'Inf_gain_ratio':[info_gain.info_gain_ratio(X,y)]
    })
    
    return(ft2)


# In[160]:


def logistic_regression(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    y=df[len(a)]
    X = df.drop(len(a),1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    predictionsLR=model.predict(x_test)
    a1=accuracy_score(y_test,predictionsLR)
    return(a1)


# In[155]:


df_all=[]
df_all=pd.DataFrame(df_all)
files_to_read = [("Combined1.arff","FileName"),("Combined2.arff","FileName"),("Combined3.arff","FileName"),("Combined5a.arff","FileName"),("Combined5b.arff","FileName")]
for (file, file_data) in files_to_read:
    df1=[]
    with open (file) as f:
        decoder=arff.ArffDecoder()
        datadictionary=decoder.decode(f,encode_nominal=True,return_type=arff.LOD)
        data=datadictionary['data']
        arff.ArffDecoder
        df1=pd.DataFrame(data)
        #df[3094] = np.where(df[3094]==1.0, 'human', 'worm')
        df1=df1.replace(-np.inf,np.nan)
        df1.fillna(df.mean(),inplace=True)
        b=extract_metafeature(df1)
        b=pd.DataFrame(b)
        df_all = df_all.append(b)
        print(df_all)


# In[156]:


print (df_all)


# In[167]:


with open ("Combined5a.arff") as f:
    decoder=arff.ArffDecoder()
    dataDictionary=decoder.decode(f,encode_nominal=True,return_type=arff.LOD)
    data=dataDictionary['data']
arff.ArffDecoder
df=pd.DataFrame(data)
logistic_regression(df)


# In[ ]:





# In[107]:


with open('result.csv', 'a') as f:
    b.to_csv(f, header=False)


# In[ ]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
predictionsLR=model.predict(x_test)
a1=accuracy_score(y_test,predictionsLR)
print (a1)


# In[ ]:


#SVM
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
predictionsSVC= svclassifier.predict(x_test)
a2=accuracy_score(y_test,predictionsSVC)
print (a2)


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)# 5 is our choice
knn.fit(x_train, y_train)
predictionsKNN = knn.predict(x_test)
a3=accuracy_score(y_test,predictionsKNN)
print (a3)


# In[ ]:


a=np.mean([a1,a2,a3])
print (a)


# In[ ]:


st=np.std([a1,a2,a3])
print(st)


# In[ ]:




