from sklearn.datasets import fetch_openml

df = fetch_openml(data_id=41214, as_frame=True).frame

print(df.shape)
print(df.head())

sum_e=df['Exposure'].sum()
sum_s=df['ClaimNb'].sum()
f=sum_s/sum_e
print (f)

g=df.groupby("Region", observed=True)[["ClaimNb","Exposure"]].sum()
g["Freq"]=g["ClaimNb"]/g["Exposure"]
print (g)

df[["ClaimNb","Exposure"]].describe()
df["Exposure"]=df["Exposure"].clip(upper=1)
df["ClaimNb"]=df["ClaimNb"].clip(upper=4)
df[["Exposure","ClaimNb"]].describe()
(df["ClaimNb"]==0).mean()

from sklearn.metrics import mean_poisson_deviance

print(mean_poisson_deviance([0, 1], [0.10, 0.10]))
print(mean_poisson_deviance([0, 1], [0.05, 0.05]))

import pandas as pd
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,train_size=0.8,random_state=42)

import numpy as np
print(np.shape(train),np.shape(test))
f_train=(train["ClaimNb"].sum()/train["Exposure"].sum())
f_test=(test["ClaimNb"].sum()/test["Exposure"].sum())

print(f_train,f_test)

from sklearn.linear_model import PoissonRegressor
X_train=pd.get_dummies(train[['Area','VehGas']],drop_first=True,dtype=float)
y_train=train['ClaimNb']/train['Exposure']
glm=PoissonRegressor(alpha=0)
glm.fit(X_train,y_train,sample_weight=train['Exposure'])
print(pd.Series(np.exp(glm.coef_),index=X_train.columns))
print(np.exp(glm.intercept_))

from sklearn.metrics import mean_poisson_deviance


X_test=pd.get_dummies(test[['Area','VehGas']],drop_first=True,dtype=float)
pred_glm=glm.predict(X_test)
pred_naif=np.full(len(test),f_train)

real_f=test['ClaimNb']/test['Exposure']
mean_poisson_deviance(real_f,pred_naif,sample_weight=test['Exposure'])
mean_poisson_deviance(real_f,pred_glm,sample_weight=test['Exposure'])

train['ageTranche']=pd.cut(train['DrivAge'],bins=[17,21,25,30,40,50,60,70,100])

trainat=train.groupby("ageTranche",observed=True)[["ClaimNb","Exposure"]].sum()
trainat['f']=trainat['ClaimNb']/trainat['Exposure']
print(trainat['f'])

test['ageTranche']=pd.cut(test['DrivAge'],bins=[17,21,25,30,40,50,60,70,100])
X_test2=pd.get_dummies(test[['Area','VehGas','ageTranche']],drop_first=True,dtype=float)
from sklearn.linear_model import PoissonRegressor
gnl=PoissonRegressor(alpha=0)
X_train2=pd.get_dummies(train[['Area','VehGas','ageTranche']],drop_first=True,dtype=float)

pred_glm2=gnl.fit(X_train2,y_train,sample_weight=train['Exposure']).predict(X_test2)

from sklearn.model_selection import mean_poisson_deviance
mean_poisson_deviance(real_f,pred_glm2,sample_weight=test['Exposure'])

mean_poisson_deviance(real_f,pred_naif,sample_weight=test['Exposure'])
mean_poisson_deviance(real_f,pred_glm,sample_weight=test['Exposure'])