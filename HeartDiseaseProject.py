
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("heart.csv")


# In[3]:


data.head(10)


# In[4]:


data["sex"].value_counts()


# In[5]:


data["fbs"].value_counts()


# In[6]:


data["exang"].value_counts()


# In[7]:


data["restecg"].value_counts()


# In[8]:


data["thal"].value_counts()


# In[9]:


data["ca"].value_counts()


# In[10]:


data["slope"].value_counts()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data["target"].value_counts()


# In[14]:


sns.countplot(x="target",data=data)


# In[30]:


sns.countplot(x="sex",data=data)


# In[31]:


sns.countplot(x="fbs",data=data)


# In[32]:


sns.countplot(x="restecg",data=data)


# In[33]:


sns.countplot(x="slope",data=data)


# In[34]:


sns.countplot(x="thal",data=data)


# In[35]:


sns.countplot(x="cp",data=data)


# In[36]:


sns.countplot(x="ca",data=data)


# In[37]:


sns.distplot(data["age"])


# In[38]:


sns.distplot(data["trestbps"])


# In[39]:


sns.distplot(data["chol"])


# In[44]:


plt.figure(figsize=(20,12))
sns.heatmap(data.corr(),cmap="coolwarm",annot=True)


# In[47]:


plt.figure(figsize=(20,12))
data.hist()


# In[48]:


data.columns


# In[ ]:


data=pd.get_dummies(data,columns=["cp","restecg","slope","ca","thal"],drop_first=True)


# In[17]:


data.head()


# In[51]:


data.columns


# In[19]:


cols=["age","trestbps","chol","thalach","oldpeak"]


# In[20]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data[cols]=sc.fit_transform(data[cols])


# In[21]:


data.head()


# In[22]:


X=data.drop("target",axis=1)


# In[23]:


Y=data["target"].values


# In[24]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[26]:


scores=[]
for k in range(3,16):
    knn=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn,X,Y,cv=10)
    scores.append(score.mean())


# In[27]:


max(scores)


# In[28]:


plt.figure(figsize=(10,6))
plt.plot(list(range(3,16)),scores,label="score")
plt.legend()


# In[29]:


kNN=KNeighborsClassifier(n_neighbors=7)
score=cross_val_score(kNN,X,Y,cv=10)


# In[30]:


score.mean()


# In[31]:


score


# In[32]:


kNN.fit(xtrain,ytrain)


# In[33]:


ypred=kNN.predict(xtest)


# In[34]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
print(accuracy_score(ytest,ypred))


# In[35]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()


# In[41]:


from sklearn.grid_search import GridSearchCV
params={"n_estimators":[10,50,100,130,150],"max_features":["auto","sqrt",0.5,0.4]}
grid=GridSearchCV(RandomForestClassifier(),params,verbose=3)


# In[58]:


grid.fit(xtrain,ytrain)


# In[59]:


grid.best_params_


# In[60]:


grid.best_estimator_


# In[61]:


ypred3=grid.predict(xtest)


# In[62]:


print(confusion_matrix(ytest,ypred3))
print(classification_report(ytest,ypred3))
print(accuracy_score(ytest,ypred3))


# In[79]:


from xgboost import XGBClassifier


# In[81]:


xgb=XGBClassifier()
xgb.fit(xtrain,ytrain)


# In[82]:


ypredx=xgb.predict(xtest)


# In[83]:


print(confusion_matrix(ytest,ypredx))
print(classification_report(ytest,ypredx))
print(accuracy_score(ytest,ypredx))


# In[75]:


model=RandomForestClassifier(n_estimators=130,max_features="sqrt")


# In[76]:


model.fit(xtrain,ytrain)


# In[77]:


ypred3=model.predict(xtest)


# In[78]:


print(confusion_matrix(ytest,ypred3))
print(classification_report(ytest,ypred3))
print(accuracy_score(ytest,ypred3))

