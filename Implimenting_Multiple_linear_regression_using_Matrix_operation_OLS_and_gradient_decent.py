#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[4]:


df = pd.read_csv(r'C:\Users\prach_sxw8up\Downloads\amsPredictionSheet1-201009-150447.csv')
df.head()


# In[5]:


df.describe()


# In[6]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[8]:


endog = df['ESE']
exog = sm.add_constant(df[['MSE','Attendance','HRS']])
print(exog)


# In[10]:


x = exog.to_numpy()
y = endog.to_numpy()
s1_xt =np.transpose(x)
print(s1_xt)


# In[11]:


s2_mul1= np.matmul(s1_xt,x)
print(s2_mul1)


# In[12]:


s3_inv=np.linalg.inv(s2_mul1)
print(s3_inv)


# In[13]:


s4_mul= np.matmul(s3_inv,s1_xt)
print(s4_mul)


# In[14]:


s5_res =np.matmul(s4_mul,y)
print(s5_res)


# In[15]:


mod = sm.OLS(endog, exog)
results = mod.fit()
print(results.summary())


# In[17]:


from sklearn import linear_model
x = df[['MSE','Attendance','HRS']]
y = df['ESE']

lm = linear_model.LinearRegression()
model = lm.fit(x,y)
lm.coef_

