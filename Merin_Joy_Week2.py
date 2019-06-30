#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd


# In[2]:


conn=sqlite3.connect('/Users/Merin/Desktop/HAP880/Assignment2/testClaims_hu.db') # enter full path here
df=pd.read_sql('select * from highUtilizationPredictionV2wco',conn)


# In[3]:


df=df.join(pd.get_dummies(df.race))


# In[4]:


df.head(4)


# In[5]:


cols=list(df.columns)


# In[6]:


cols.remove('index')
cols.remove('race')
cols.remove('patient_id')
cols.remove('HighUtilizationY2')
cols.remove('claimCount')


# In[35]:


sz=df.index.size


# In[8]:


from sklearn.utils import shuffle


# In[9]:


df=shuffle(df)


# In[10]:


import numpy as np
rnd=np.random.rand(1,sz)
df['rnd']=list(rnd[0])
df=df.sort_values('rnd')


# In[11]:


df.head(5)


# In[12]:


# split to training and testing
tr=df[:int(sz*0.8)] 
ts=df[int(sz*0.8):]


# ### 1. Create a plot that shows performance (AUC) of random forest models (x axis) with 10, 20, 30, … 200 (y axis) on training and testing data (build all 20 models to complete the assignment). Use highUtilizationPredictionV2wco dataset.

# In[13]:


from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# In[14]:


rf = dict()
probs_rf = dict()
fpr_rf = dict()
tpr_rf = dict() 
thresholds_rf = dict()
auc_rf = dict()


# In[15]:


for i in range(10, 201, 10):
    rf[i]=RandomForestClassifier(n_estimators =i)
    rf[i].fit(tr[cols],tr['HighUtilizationY2'])
    probs_rf[i]=rf[i].predict_proba(ts[cols])
    fpr_rf[i], tpr_rf[i], thresholds_rf[i] = roc_curve(ts['HighUtilizationY2'],probs_rf[i][:,1])
    auc_rf[i]=auc(fpr_rf[i],tpr_rf[i])
    plt.title('Random Forest')
    plt.plot(fpr_rf[i], tpr_rf[i],label = "{} trees".format(i)+" (AUC= "+str(round((auc_rf[i]),3))+")")
    plt.legend(loc='center right', bbox_to_anchor=(1,0.5,0.5,0.5))


# In[29]:


i_rf = []
auc_rf_plot = []
for i in range(10, 201, 10):
    i_rf.append(i)
    auc_rf_plot.append(auc_rf[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Number of trees')
plt.plot(i_rf,auc_rf_plot)


# ### 2. Rank input attributes in the data. Create a plot that shows dependency between number of selected top attributes (x axis) and AUC of learned model (y axis). You can use any attribute selection method and any classifier you want.Use highUtilizationPredictionV2wco dataset.

# In[22]:


from sklearn.feature_selection import mutual_info_classif
mic = mutual_info_classif(tr[cols],tr['HighUtilizationY2']) # this is one of measures


# In[24]:


s=pd.DataFrame()
s['att']=cols
s['mic']=mic
s.head()


# In[25]:


rf100=RandomForestClassifier(n_estimators=100)
auc_rf100_plot_mic = dict()
for i in range(5, 51, 5):
    cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:i]
    rf100.fit(tr[cols_sel_mic],tr['HighUtilizationY2'])
    probs_rf100=rf100.predict_proba(ts[cols_sel_mic])
    fpr_rf100, tpr_rf100, thresholds_rf100 = roc_curve(ts['HighUtilizationY2'],probs_rf100[:,1])
    auc_rf100=auc(fpr_rf100,tpr_rf100)
    auc_rf100_plot_mic[i]=auc_rf100
    plt.title('Random Forest')
    plt.plot(fpr_rf100, tpr_rf100,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_rf100),3))+")")
    plt.legend(loc='best')


# In[30]:


i_rf100 =[]
auc_rf100_plot2 = []
for i in range(5, 51, 5):
    i_rf100.append(i)
    auc_rf100_plot2.append(auc_rf100_plot_mic[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using mic')
plt.plot(i_rf100,auc_rf100_plot2)


# ### 3. Create a learning curve for the data to check AUC (y axis) based on size of data (x axis).

# In[27]:


rf100=RandomForestClassifier(n_estimators=100)
sz=tr.index.size
auc_rf100_percent = dict()
for i in range (10, 101, 10):
    tt=tr[:int(sz*i/100.0)]
    rf100.fit(tt[cols],tt['HighUtilizationY2'])
    probs_rf100=rf100.predict_proba(ts[cols])
    fpr_rf100, tpr_rf100, thresholds_rf100 = roc_curve(ts['HighUtilizationY2'],probs_rf100[:,1])
    auc_rf100=auc(fpr_rf100,tpr_rf100)
    auc_rf100_percent[i] =auc_rf100
    plt.title('Random Forest')
    plt.plot(fpr_rf100, tpr_rf100, label = " {} % of train data".format(i)+" (AUC= "+str(round((auc_rf100),3))+")")
    plt.legend(loc='best')


# In[31]:


i_rf100 =[]
auc_rf100_plot_per = []
for i in range(10, 101, 10):
    i_rf100.append(i)
    auc_rf100_plot_per.append(auc_rf100_percent[i])
plt.title('Learning curve')
plt.ylabel('AUC')
plt.xlabel('Size of Data(%)')
plt.plot(i_rf100,auc_rf100_plot_per)


# ### 4. Create 5 more learning curves (on one plot) for different numbers of input attributes. Do it for random forest, logistic regression, and naïve Bayes.

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# In[33]:


rf100=RandomForestClassifier(n_estimators=100)
lr = LogisticRegression()
nb = GaussianNB()


# ### a. Use random selection of input attributes

# In[44]:


sz=df.index.size
tr=df[:int(sz*0.8)] 
ts=df[int(sz*0.8):]


# In[50]:


import random
tr_size=tr.index.size
for j in range(5):
    auc_rf_percent = dict()
    i_rf =[]
    auc_rf_plot_percent = []
    random.shuffle(cols)
    attri=random.randint(10,len(cols))
    rand_cols=cols[ :attri]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        rf100.fit(tt[rand_cols],tt['HighUtilizationY2'])
        probs_rf100=rf100.predict_proba(ts[rand_cols])
        fpr_rf100, tpr_rf100, thresholds_rf100 = roc_curve(ts['HighUtilizationY2'],probs_rf100[:,1])
        auc_rf100=auc(fpr_rf100,tpr_rf100)
        auc_rf_percent[i] =auc_rf100
        i_rf.append(i)
        auc_rf_plot_percent.append(auc_rf_percent[i])
    plt.title('Random Forest - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_rf,auc_rf_plot_percent,label = " {} random attributes".format(attri))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# In[52]:


tr_size=tr.index.size
for j in range(5):
    auc_lr_percent = dict()
    i_lr =[]
    auc_lr_plot_percent = []
    random.shuffle(cols)
    attri=random.randint(10,len(cols))
    rand_cols=cols[ :attri]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        lr.fit(tt[rand_cols],tt['HighUtilizationY2'])
        probs_lr=lr.predict_proba(ts[rand_cols])
        fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['HighUtilizationY2'],probs_lr[:,1])
        auc_lr=auc(fpr_lr,tpr_lr)
        auc_lr_percent[i] =auc_lr
        i_lr.append(i)
        auc_lr_plot_percent.append(auc_lr_percent[i])
    plt.title('Logistic Regression - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_lr,auc_lr_plot_percent,label = " {} random attributes".format(attri))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# In[53]:


tr_size=tr.index.size
for j in range(5):
    auc_nb_percent = dict()
    i_nb =[]
    auc_nb_plot_percent = []
    random.shuffle(cols)
    attri=random.randint(10,len(cols))
    rand_cols=cols[ :attri]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        nb.fit(tt[rand_cols],tt['HighUtilizationY2'])
        probs_nb=nb.predict_proba(ts[rand_cols])
        fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['HighUtilizationY2'],probs_nb[:,1])
        auc_nb=auc(fpr_nb,tpr_nb)
        auc_nb_percent[i] =auc_nb
        i_nb.append(i)
        auc_nb_plot_percent.append(auc_nb_percent[i])
    plt.title('Naive Bayes - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_nb,auc_nb_plot_percent,label = " {} random attributes".format(attri))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# ### b. Use ranking of attributes from question 2

# In[55]:


tr_size=tr.index.size
for j in range(10,51,10):
    auc_rf_percent = dict()
    i_rf =[]
    auc_rf_plot_percent = []
    cols_sel_mic=s.sort_values('mic', ascending=False)['att'][:j]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        rf100.fit(tt[cols_sel_mic],tt['HighUtilizationY2'])
        probs_rf100=rf100.predict_proba(ts[cols_sel_mic])
        fpr_rf100, tpr_rf100, thresholds_rf100 = roc_curve(ts['HighUtilizationY2'],probs_rf100[:,1])
        auc_rf100=auc(fpr_rf100,tpr_rf100)
        auc_rf_percent[i] =auc_rf100
        i_rf.append(i)
        auc_rf_plot_percent.append(auc_rf_percent[i])
    plt.title('Random Forest - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_rf,auc_rf_plot_percent,label = "Top"+" {} attributes".format(j))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# In[56]:


tr_size=tr.index.size
for j in range(10,51,10):
    auc_lr_percent = dict()
    i_lr =[]
    auc_lr_plot_percent = []
    cols_sel_mic=s.sort_values('mic', ascending=False)['att'][:j]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        lr.fit(tt[cols_sel_mic],tt['HighUtilizationY2'])
        probs_lr=lr.predict_proba(ts[cols_sel_mic])
        fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['HighUtilizationY2'],probs_lr[:,1])
        auc_lr=auc(fpr_lr,tpr_lr)
        auc_lr_percent[i] =auc_lr
        i_lr.append(i)
        auc_lr_plot_percent.append(auc_lr_percent[i])
    plt.title('Logistic Regression - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_lr,auc_lr_plot_percent,label = "Top"+" {} attributes".format(j))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# In[59]:


tr_size=tr.index.size
for j in range(10,51,10):
    auc_nb_percent = dict()
    i_nb =[]
    auc_nb_plot_percent = []
    cols_sel_mic=s.sort_values('mic', ascending=False)['att'][:j]
    for i in range (10, 101, 10):
        tt=tr[:int(tr_size*i/100.0)]
        nb.fit(tt[cols_sel_mic],tt['HighUtilizationY2'])
        probs_nb=nb.predict_proba(ts[cols_sel_mic])
        fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['HighUtilizationY2'],probs_nb[:,1])
        auc_nb=auc(fpr_nb,tpr_nb)
        auc_nb_percent[i] =auc_nb
        i_nb.append(i)
        auc_nb_plot_percent.append(auc_nb_percent[i])
    plt.title('Naive Bayes - Learning curves')
    plt.ylabel('AUC')
    plt.xlabel('Size of Data(%)')
    plt.plot(i_nb,auc_nb_plot_percent,label = "Top"+" {} attributes".format(j))
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))

