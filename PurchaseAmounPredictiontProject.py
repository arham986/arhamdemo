
# coding: utf-8

# In[134]:


from google.colab import files
uploaded=files.upload()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("train.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[7]:


sns.countplot(x="Gender",data=data)


# In[8]:


sns.countplot(x="Age",data=data)


# In[9]:


sns.set_style("darkgrid")
sns.distplot(data["Purchase"],bins=20,kde=False)


# In[10]:


sns.countplot(x="Marital_Status",data=data)


# In[11]:


data["Occupation"].nunique()


# In[12]:


sns.countplot(x="City_Category",data=data)


# In[13]:


plt.figure(figsize=(10,8))
sns.boxplot(x="Age",y="Purchase",data=data)


# In[14]:


plt.figure(figsize=(10,8))
sns.barplot(x="Age",y="Purchase",data=data)


# In[15]:


data["Product_Category_1"].nunique()


# In[16]:


data["Product_Category_2"].nunique()


# In[17]:


data["Product_Category_3"].nunique()


# In[18]:


data["Product_Category_2"].value_counts(dropna=False)


# In[19]:


data["Product_Category_3"].value_counts(dropna=False)


# In[6]:


data["Product_Category_2"].fillna(value=8.0,inplace=True)


# In[7]:


data["Product_Category_3"].fillna(value=16.0,inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


data["Gender"]=data["Gender"].map({"F":0,"M":1})


# In[10]:


data["Age"].value_counts()


# In[11]:


data["Age"]=data["Age"].map({"0-17":0,"18-25":1,"26-35":2,"36-45":3,"46-50":4,"51-55":5,"55+":6})


# In[12]:


df=pd.get_dummies(data["Product_Category_2"])


# In[62]:


dff=pd.get_dummies(data["Product_Category_2"])


# In[63]:


dff.head()


# In[13]:


df=df.loc[0:,[8.0,14.0,2.0,16.0,15.0]]


# In[29]:


df.head()


# In[14]:


df2=pd.get_dummies(data["Product_Category_3"])


# In[64]:


dff2=pd.get_dummies(data["Product_Category_3"])


# In[65]:


dff2.head()


# In[15]:


df2=df2.loc[0:,[16.0,15.0,4.0]]


# In[33]:


df2.head()


# In[16]:


df3=pd.get_dummies(data["City_Category"],drop_first=True)


# In[35]:


df3.head()


# In[17]:


data.shape


# In[18]:


d=pd.concat([data[["User_ID","Product_ID","Age","Gender","Occupation","Stay_In_Current_City_Years","Marital_Status","Purchase"]],df,df2,df3],axis=1)


# In[66]:


ds=pd.concat([data[["User_ID","Product_ID","Age","Gender","Occupation","Stay_In_Current_City_Years","Marital_Status","Purchase"]],dff,dff2,df3],axis=1)


# In[38]:


d.head()


# In[67]:


ds.shape


# In[19]:


df4=pd.get_dummies(data["Product_Category_1"],drop_first=True)


# In[20]:


df4.head()


# In[21]:


d=pd.concat([d,df4],axis=1)


# In[68]:


ds=pd.concat([ds,df4],axis=1)


# In[42]:


d.head()


# In[22]:


d.shape


# In[23]:


d["Stay_In_Current_City_Years"].value_counts()


# In[69]:


ds["Stay_In_Current_City_Years"]=ds["Stay_In_Current_City_Years"].map({"0":0,"1":1,"2":2,"3":3,"4+":4})


# In[24]:


d["Stay_In_Current_City_Years"]=d["Stay_In_Current_City_Years"].map({"0":0,"1":1,"2":2,"3":3,"4+":4})


# In[46]:


d.head()


# In[25]:


X=d.drop("Purchase",axis=1)


# In[73]:


Xs=ds.drop("Purchase",axis=1)


# In[26]:


X.shape


# In[27]:


X=X.iloc[0:,0:].values


# In[74]:


Xs=Xs.iloc[0:,0:].values


# In[28]:


X.shape


# In[75]:


Xs.shape


# In[29]:


X=X[0:,2:]


# In[77]:


Xs=Xs[0:,2:]


# In[30]:


X.shape


# In[78]:


Xs.shape


# In[31]:


Y=d["Purchase"].values


# In[33]:


Y


# In[39]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[79]:


from sklearn.model_selection import train_test_split
xtrains,xtests,ytrains,ytests=train_test_split(Xs,Y,test_size=0.3,random_state=0)


# In[56]:


from xgboost import XGBRegressor
model=XGBRegressor()
model.fit(xtrain,ytrain)


# In[80]:


from xgboost import XGBRegressor
model=XGBRegressor()
model.fit(xtrains,ytrains)


# In[ ]:


ypred=model.predict(xtest)


# In[81]:


ypreds=model.predict(xtests)


# In[58]:


from sklearn.metrics import mean_squared_error
import numpy as np
print("RMSE:",np.sqrt(mean_squared_error(ytest,ypred)))


# In[82]:


print("RMSE:",np.sqrt(mean_squared_error(ytests,ypreds)))


# In[ ]:


params={"learning_rate":[0.05,0.1,0.2,0.3],"gamma":[0,0.2,0.1,0.4],"max_depth":[3,5,6,7,9],"subsample":[0.5,0.7,1],"colsample_bytree":[0.4,0.5,0.7,1],"min_child_weight":[1,2,3,4]}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
rsv=RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=params,n_iter=50,n_jobs=-1,verbose=3,cv=5)


# In[61]:


rsv.fit(xtrain,ytrain)


# In[62]:


rsv.best_estimator_


# In[83]:


xgb=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.3, max_delta_step=0,
             max_depth=9, min_child_weight=4, missing=None, n_estimators=200,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.7, verbosity=1)


# In[84]:


xgb.fit(xtrains,ytrains)


# In[85]:


ypreds1=xgb.predict(xtests)


# In[ ]:


ypred=xgb.predict(xtest)


# In[66]:


print("RMSE:",np.sqrt(mean_squared_error(ytest,ypred))


# In[86]:


print("RMSE:",np.sqrt(mean_squared_error(ytests,ypreds1)))


# In[ ]:


data1=pd.read_csv("test.csv")


# In[98]:


data1.head()


# In[99]:


data1.isnull().sum()


# In[ ]:


data1["Gender"]=data1["Gender"].map({"F":0,"M":1})


# In[ ]:


data1["Age"]=data1["Age"].map({"0-17":0,"18-25":1,"26-35":2,"36-45":3,"46-50":4,"51-55":5,"55+":6})


# In[102]:


data1.head()


# In[ ]:


data1["Product_Category_2"].fillna(value=8.0,inplace=True)


# In[ ]:


data1["Product_Category_3"].fillna(value=16.0,inplace=True)


# In[105]:


data1.isnull().sum()


# In[ ]:


daf=pd.get_dummies(data1["Product_Category_2"])


# In[107]:


daf.head()


# In[ ]:


daf=daf.loc[0:,[8.0,14.0,2.0,16.0,15.0]]


# In[109]:


daf.head()


# In[ ]:


daf2=pd.get_dummies(data1["Product_Category_3"])


# In[ ]:


daf2=daf2.loc[0:,[16.0,15.0,4.0]]


# In[112]:


daf2.head()


# In[ ]:


daf3=pd.get_dummies(data1["City_Category"],drop_first=True)


# In[114]:


daf3.head()


# In[ ]:


d1=pd.concat([data1[["User_ID","Product_ID","Age","Gender","Occupation","Stay_In_Current_City_Years","Marital_Status"]],daf,daf2,daf3],axis=1)


# In[116]:


d1.head()


# In[ ]:


daf4=pd.get_dummies(data1["Product_Category_1"],drop_first=True)


# In[118]:


daf4.head()


# In[ ]:


daf4[19]=0


# In[ ]:


daf4[20]=0


# In[121]:


daf4.head()


# In[ ]:


d1=pd.concat([d1,daf4],axis=1)


# In[123]:


d1.head()


# In[124]:


d1.shape


# In[ ]:


d1["Stay_In_Current_City_Years"]=d1["Stay_In_Current_City_Years"].map({"0":0,"1":1,"2":2,"3":3,"4+":4})


# In[126]:


d1.dtypes


# In[ ]:


Xtest=d1.drop(["User_ID","Product_ID"],axis=1)


# In[128]:


Xtest.shape


# In[ ]:


Xtest=Xtest.iloc[0:,:].values


# In[130]:


Xtest


# In[131]:


Xtest.shape


# In[ ]:


ypred2=xgb.predict(Xtest)


# In[ ]:


sample=pd.read_csv("sample_submission_LMg97w5.csv")


# In[136]:


sample.head()


# In[139]:


type(sample["Comb"][0])


# In[140]:


sample.shape


# In[141]:


data1.shape


# In[142]:


data1.head()


# In[ ]:


sample["Purchase"]=50


# In[144]:


sample.head()


# In[148]:


len(ypred2)


# In[149]:


sample.shape


# In[ ]:


sample["Purchase"]=ypred2


# In[151]:


sample.head(10)


# In[ ]:


from google.colab import files


# In[177]:


sample.head(10)


# In[ ]:


sample.to_csv("final.csv",index=None)


# In[ ]:


files.download("final.csv")


# In[ ]:


sam=pd.read_csv("sample_submission_LMg97w5.csv")


# In[176]:


sam.head()


# In[ ]:


df=data1[["User_ID","Product_ID"]]


# In[189]:


df.head()


# In[190]:


df["Purchase"]=ypred2


# In[192]:


df.head(10)


# In[ ]:


df.to_csv("sub.csv",index=None)


# In[ ]:


files.download("sub.csv")


# In[199]:


X.shape


# In[200]:


len(Y)


# In[201]:


xgb.fit(X,Y)


# In[ ]:


ypred3=xgb.predict(Xtest)


# In[204]:


len(ypred3)


# In[205]:


df["Purchase"]=ypred3


# In[206]:


df.head()


# In[ ]:


df.to_csv("subms.csv",index=None)


# In[ ]:


files.download("subms.csv")


# In[34]:


from sklearn.model_selection import RandomizedSearchCV


# In[35]:


from sklearn.ensemble import RandomForestRegressor


# In[40]:


rf=RandomForestRegressor(n_estimators=1500,max_depth=10)


# In[37]:


param={"n_estimators":[1000,2000,2500],"max_features":["auto",0.7,0.5],"max_depth":[9,10,11]}


# In[38]:


rsv1=RandomizedSearchCV(estimator=RandomForestRegressor(),param_distributions=param,n_iter=20,n_jobs=-1,verbose=3,cv=4)


# In[41]:


rf.fit(xtrain,ytrain)


# In[42]:


ypred4=rf.predict(xtest)


# In[44]:


from sklearn.metrics import mean_squared_error


# In[45]:


print("RMSE:",np.sqrt(mean_squared_error(ytest,ypred4)))


# In[47]:


from xgboost import XGBRegressor


# In[96]:


xgb1=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.3, max_delta_step=0,
             max_depth=7, min_child_weight=4, missing=None, n_estimators=160,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=1, reg_lambda=0, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.7, verbosity=1)


# In[97]:


xgb1.fit(xtrain,ytrain)


# In[98]:


ypred5=xgb1.predict(xtest)


# In[99]:


print("RMSE:",np.sqrt(mean_squared_error(ytest,ypred5)))

