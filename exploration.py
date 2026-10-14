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

from sklearn.metrics import mean_poisson_deviance
mean_poisson_deviance(real_f,pred_glm2,sample_weight=test['Exposure'])

mean_poisson_deviance(real_f,pred_naif,sample_weight=test['Exposure'])
mean_poisson_deviance(real_f,pred_glm,sample_weight=test['Exposure'])

def preparer(data):
    X=pd.DataFrame(index=data.index)
    X['LogDensity']=np.log(data['Density'])
    Tranche=pd.cut(data['DrivAge'],bins=[17,21,25,30,40,50,60,70,100])
    age=pd.get_dummies(Tranche,prefix='Age',drop_first=True,dtype=float)
    X=pd.concat([X,age,pd.get_dummies(data[['Area','VehGas']],drop_first=True,dtype=float)],axis=1)
    X['BonusMalus']=data['BonusMalus']
    return X

X_train=preparer(train)
X_test=preparer(test)
print(list(X_train.columns)==list(X_test.columns))

from sklearn.linear_model import PoissonRegressor
gnl=PoissonRegressor(alpha=0,max_iter=1000)
pred_gnl3=gnl.fit(X_train,y_train,sample_weight=train['Exposure']).predict(X_test)

from sklearn.metrics import mean_poisson_deviance
a=mean_poisson_deviance(real_f,pred_gnl3,sample_weight=test['Exposure'])
b=mean_poisson_deviance(real_f,pred_naif,sample_weight=test['Exposure'])
print(1-a/b)
print(a)

train_sub,valid=train_test_split(train,train_size=0.8,random_state=42)
X_sub=preparer(train_sub)
X_valid=preparer(valid)
f_sub=train_sub['ClaimNb'].sum()/train_sub['Exposure'].sum()
f_valid=valid['ClaimNb'].sum()/valid['Exposure'].sum()
y_sub=train_sub['ClaimNb']/train_sub['Exposure']
y_valid=valid['ClaimNb']/valid['Exposure']

groupes=["LogDensity", "BonusMalus", "Age_", "Area_", "VehAge_"]

for g in groupes:
    X_sub_sans=X_sub.drop(columns=[c for c in X_sub.columns if c.startswith(g)])
    X_valid_sans=X_valid.drop(columns=[c for c in X_valid.columns if c.startswith(g)])
    a=gnl.fit(X_sub_sans,y_sub,sample_weight=train_sub['Exposure']).predict(X_valid_sans)
    b=mean_poisson_deviance(y_valid,a,sample_weight=valid['Exposure'])
    print("deviance=",b)
    f=train_sub['ClaimNb'].sum()/train_sub['Exposure'].sum()
    pred_naif2=np.full(len(valid),f)
    b2=mean_poisson_deviance(y_valid,pred_naif2,sample_weight=valid['Exposure'])
    print("part de la deviance expliquée=",1-b/b2)
    print(g, X_sub.shape[1] - X_sub_sans.shape[1], "colonnes retirées")

print("deviance naive=",b2)

mean_poisson_deviance(y_valid,gnl.fit(X_sub,y_sub,sample_weight=train_sub['Exposure']).predict(X_valid),sample_weight=valid['Exposure'])

h=pd.cut(train['VehAge'],bins=[-1,0,2,5,10,15,100])
train2=pd.concat([train,h],axis=1)
p=train2.groupby(h)[["ClaimNb",'Exposure']].sum()
p['f']=p['ClaimNb']/p['Exposure']
print(p)

def preparer(data):
    X=pd.DataFrame(index=data.index)
    X['LogDensity']=np.log(data['Density'])
    Tranche=pd.cut(data['DrivAge'],bins=[17,21,25,30,40,50,60,70,100])
    Tranche2=pd.cut(data['VehAge'],bins=[-1,0,2,5,10,15,100])
    VehAge=pd.get_dummies(Tranche2,prefix='VehAge',drop_first=True,dtype=float)
    age=pd.get_dummies(Tranche,prefix='Age',drop_first=True,dtype=float)
    X=pd.concat([X,age,VehAge,pd.get_dummies(data[['Area']],drop_first=True,dtype=float)],axis=1)
    X['BonusMalus']=data['BonusMalus']
    return X

gnlA=PoissonRegressor(alpha=0,max_iter=1000)
pred_gnlA=gnlA.fit(X_sub,y_sub,sample_weight=train_sub['Exposure']).predict(X_valid)
gnlB=PoissonRegressor(alpha=0,max_iter=1000)
pred_gnlB=gnlB.fit(X_sub.drop(columns=[c for c in X_sub.columns if c.startswith('VehAge')]),y_sub,sample_weight=train_sub['Exposure']).predict(X_valid.drop(columns=[c for c in X_valid.columns if c.startswith('VehAge')]))

a=mean_poisson_deviance(y_valid,pred_gnlA,sample_weight=valid['Exposure'])
b=mean_poisson_deviance(y_valid,pred_gnlB,sample_weight=valid['Exposure'])
print(a)
print(b)
