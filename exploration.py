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