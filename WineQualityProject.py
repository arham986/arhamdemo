
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


data=pd.read_csv(r"C:\Users\Lenovo\Downloads\wine.csv")


# In[54]:


data.head()


# In[93]:


X=data.drop("Wine",axis=1)


# In[94]:


X.head()


# In[95]:


X=X.iloc[0:,:].values
X


# In[96]:


X.shape


# In[91]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)


# In[98]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xstd=sc.fit_transform(X)


# In[103]:


pcax=pca.fit_transform(xstd)


# In[104]:


pcax.shape


# In[106]:


plt.scatter(pcax[0:,0],pcax[0:,1],c=data["Wine"])


# In[84]:


XX=X[0:,[0,2,4,5,6,9,10,11,12]]


# In[108]:


from sklearn.model_selection import train_test_split
xtrain2,xtest2,ytrain2,ytest2=train_test_split(pcax,Y,test_size=0.3,random_state=100)


# In[109]:


from sklearn.tree import DecisionTreeClassifier
dt1=DecisionTreeClassifier()
dt1.fit(xtrain2,ytrain2)


# In[85]:


XX


# In[110]:


ypred3=dt1.predict(xtest2)


# In[111]:


print(classification_report(ytest,ypred3))
print(accuracy_score(ytest,ypred3))


# In[86]:


from sklearn.model_selection import train_test_split
xtrain1,xtest1,ytrain1,ytest1=train_test_split(XX,Y,test_size=0.3,random_state=100)


# In[87]:


dt=DecisionTreeClassifier()
dt.fit(xtrain1,ytrain1)


# In[88]:


ypred2=dt.predict(xtest1)


# In[89]:


from sklearn.metrics import accuracy_score


# In[90]:


print(classification_report(ytest,ypred2))
print(accuracy_score(ytest,ypred2))


# In[9]:


Y=data["Wine"].values


# In[10]:


Y


# In[36]:


plt.figure(figsize=(25,7))
sns.countplot(x="Alcohol",hue="Wine",data=data,palette="Set1")


# In[37]:


sns.barplot(x="Wine",y="Alcohol",data=data)


# In[39]:


sns.pairplot(data,hue="Wine",palette="Set1")


# In[61]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=100)


# In[62]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)


# In[74]:


from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()
model.fit(xtrain,ytrain)


# In[75]:


model.feature_importances_


# In[32]:


X.shape[1]


# In[29]:


ypred1=model.predict(xtest)


# In[30]:


print(classification_report(ytest,ypred1))


# In[63]:


ypred=dtree.predict(xtest)
ypred


# In[64]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypred))
print(accuracy_score(ytest,ypred))


# In[15]:


print(confusion_matrix(ytest,ypred))


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(xtrain,ytrain)


# In[17]:


rpred=rfc.predict(xtest)
rpred


# In[18]:


print(confusion_matrix(ytest,rpred))


# In[31]:


print(classification_report(ytest,rpred))


# In[112]:


xtest.shape


# In[113]:


X.shape


# In[114]:


print("Done")

