#!/usr/bin/env python
# coding: utf-8

# # Graduate Admission Prediction using ML by KODI VENU

# In[1]:


import pandas as pd
data=pd.read_csv('Admission_Predict.csv')


# Top 5 rows

# In[2]:


data.head()


# Last 5 rows

# In[3]:


data.tail()


# Dataset Shape

# In[4]:


data.shape


# In[5]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[6]:


data.info()


# Check null values in the dataset

# In[7]:


data.isnull().sum()


# Dataset Statistics

# In[8]:


data.describe()


# Drop irrelevant features

# In[9]:


data.columns


# In[10]:


data.drop('Serial No.',axis=1)


# In[11]:


data.columns


# store feature matrix in X & response (target) in vector y

# In[12]:


data.head(1)


# In[13]:


data.columns


# In[17]:


X=data.drop('Chance of Admit ',axis=1)
y=data['Chance of Admit ']


# Train/Test split

# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[19]:


y_train


# Feature scaling

# In[20]:


data.head()


# In[22]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train


# Import models

# In[23]:


data.head()


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Model Training

# In[28]:


lr=LinearRegression()
lr.fit(X_train,y_train)

svm=SVR()
svm.fit(X_train,y_train)

rf=RandomForestRegressor()
rf.fit(X_train,y_train)

gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)


# Prediction on Test Data

# In[29]:


y_pred1=lr.predict(X_test)
y_pred2=svm.predict(X_test)
y_pred3=rf.predict(X_test)
y_pred4=gr.predict(X_test)


# Evaluating the algorithm

# In[30]:


from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)


# In[31]:


print(score1,score2,score3,score4)


# In[32]:


final_data=pd.DataFrame({'Models':['LR','SVR','RF','GR'],'R2_SCORE':[score1,score2,score3,score4]})


# In[33]:


final_data


# In[34]:


import seaborn as sns


# In[38]:


sns.barplot(final_data['Models'],final_data['R2_SCORE'])


# In[39]:


data.head()


# In[40]:


import numpy as np


# In[43]:


y_train=[1 if value>0.8 else 0 for value in y_train]
y_test=[1 if value>0.8 else 0 for value in y_test]
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train


# Import Models

# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Model Training & Evaluation

# In[58]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred1=lr.predict(X_test)
print(accuracy_score(y_test,y_pred1))


# In[59]:


svm=svm.SVC()
svm.fit(X_train,y_train)
y_pred2=svm.predict(X_test)
print(accuracy_score(y_test,y_pred2))


# In[60]:


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
print(accuracy_score(y_test,y_pred3))


# In[61]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred4=rf.predict(X_test)
print(accuracy_score(y_test,y_pred4))


# In[62]:


gr=GradientBoostingClassifier()
gr.fit(X_train,y_train)
y_pred5=gr.predict(X_test)
print(accuracy_score(y_test,y_pred5))


# In[63]:


final_data=pd.DataFrame({'Models':['LR','SVC','KNN','RF','GBC'],'ACC_SCORE':[accuracy_score(y_test,y_pred1),accuracy_score(y_test,y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]})


# In[64]:


final_data


# In[65]:


import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC_SCORE'])


# Save the Model

# In[66]:


data.columns


# In[70]:


X=data.drop('Chance of Admit ',axis=1)
y=data['Chance of Admit ']


# In[71]:


y=[1 if value >0.8 else 0 for value in y]
y=np.array(y)
y


# In[72]:


X=sc.fit_transform(X)
X


# In[73]:


gr=GradientBoostingClassifier()
gr.fit(X,y)


# In[79]:


import joblib
joblib.dump(gr,'admission_model')
model=joblib.load('admission_model')


# In[80]:


model.predict(sc.transform([[337,118,4,4,5,4.5,9.65,1]]))


# In[ ]:





# In[ ]:




