from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import torch.nn as nn
import torch
torch.manual_seed(42)

df = fetch_openml(data_id=41214, as_frame=True).frame

df["Exposure"]=df["Exposure"].clip(upper=1)
df["ClaimNb"]=df["ClaimNb"].clip(upper=4)

train,test=train_test_split(df,train_size=0.8,random_state=42)
train_sub,valid=train_test_split(train,train_size=0.8,random_state=42)

f_sub=train_sub['ClaimNb'].sum()/train_sub['Exposure'].sum()
f_valid=valid['ClaimNb'].sum()/valid['Exposure'].sum()
y_sub=train_sub['ClaimNb']/train_sub['Exposure']
y_valid=valid['ClaimNb']/valid['Exposure']
e_sub = train_sub['Exposure']
e_valid = valid['Exposure']

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

X_sub_glm=preparer(train_sub)
X_valid_glm=preparer(valid)

glm=PoissonRegressor(alpha=0,max_iter=1000)
lam_glm=glm.fit(X_sub_glm,y_sub,sample_weight=train_sub['Exposure']).predict(X_valid_glm)





colonnes = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
            "Area", "VehBrand", "VehGas", "Region"]

X_sub_gb=train_sub[colonnes].copy()
X_valid_gb=valid[colonnes].copy()

X_sub_gb["VehGas"] = (X_sub_gb["VehGas"] == "'Regular'").astype(float)
X_valid_gb["VehGas"] = (X_valid_gb["VehGas"] == "'Regular'").astype(float)

gb=HistGradientBoostingRegressor(loss='poisson',max_leaf_nodes=15,max_iter=1000,
                                 early_stopping=True,n_iter_no_change=10,random_state=42)

lam_gb=gb.fit(X_sub_gb,sample_weight=e_sub,y=y_sub).predict(X_valid_gb)




k=["VehPower","VehAge",'DrivAge',"BonusMalus","LogDensity"]
l=['Area','VehBrand','Region','VehGas']

def preparer3(X,ref):
    X["LogDensity"]=np.log(X["Density"])
    ref["LogDensity"]=np.log(ref["Density"])
    partie1=pd.DataFrame({})
    for K in k:
        partie1[K]=(X[K]-ref[K].mean())/(ref[K].var()**(1/2))
    partie2=pd.get_dummies(X[l],drop_first=True,dtype=float)
    sortie=pd.concat([partie1,partie2],axis=1)
    return(sortie)

train_sub_norm=preparer3(train_sub,train_sub)
valid_norm=preparer3(valid,train_sub)

X_train_nn=torch.tensor(train_sub_norm.to_numpy(),dtype=torch.float32)
X_valid_nn=torch.tensor(valid_norm.to_numpy(),dtype=torch.float32)
E_train_nn=torch.tensor(train_sub["Exposure"].to_numpy(),dtype=torch.float32)
E_valid_nn=torch.tensor(valid['Exposure'].to_numpy(),dtype=torch.float32)
Y_train_nn=torch.tensor((train_sub['ClaimNb']/train_sub["Exposure"]).to_numpy(),dtype=torch.float32)
Y_valid_nn=torch.tensor((valid["ClaimNb"]/valid["Exposure"]).to_numpy(),dtype=torch.float32)

def perte(eta,y,E):
    eta2=eta.squeeze(1)
    debut=(nn.PoissonNLLLoss(log_input=True,reduction="none")(eta2,y)*E/E.sum()).sum()
    return (debut)

def deviance(X,data,modele):
    with torch.no_grad():
        retour=torch.exp(modele(X).squeeze(1)).numpy()
    return(mean_poisson_deviance(data["ClaimNb"]/data["Exposure"],retour,sample_weight=data["Exposure"]))

def entrainer(nb_neurones,nb_epoque):
    torch.manual_seed(42)
    modele3=nn.Sequential(
        nn.Linear(42,nb_neurones),
        nn.ReLU(),
        nn.Linear(nb_neurones,1)
    )
    optimiseur=torch.optim.Adam(modele3.parameters(),lr=0.05)
    list_valid=[]
    list_train=[]
    for epoque in range(nb_epoque):
        optimiseur.zero_grad()
        l=perte(modele3(X_train_nn),Y_train_nn,E_train_nn)
        if epoque%100==0 :
            dev=deviance(X_train_nn,train_sub,modele3)
            dev2=deviance(X_valid_nn,valid,modele3)
            list_valid.append(dev2)
            list_train.append(dev)
            print(epoque,"train=",dev,"valid=",dev2)
        l.backward()
        optimiseur.step()
    return(modele3,list_train,list_valid)

m8, ht8, hv8 = entrainer(8, 1300)
lam_nn= torch.exp(m8(X_valid_nn).squeeze(1)).detach().numpy()

print(mean_poisson_deviance(y_valid,lam_glm,sample_weight=valid['Exposure']))
print(mean_poisson_deviance(y_valid,lam_nn,sample_weight=valid['Exposure']))
print(mean_poisson_deviance(y_valid,lam_gb,sample_weight=valid['Exposure']))


y_v = y_valid.to_numpy()
e_v = e_valid.to_numpy()
n = len(y_v)
rng = np.random.default_rng(42)
diffs_glm = []

for b in range(500):
    idx = rng.integers(0, n, size=n)
    d_glm = mean_poisson_deviance(y_v[idx], lam_glm[idx], sample_weight=e_v[idx])
    d_nn  = mean_poisson_deviance(y_v[idx], lam_nn[idx],  sample_weight=e_v[idx])
    diffs_glm.append(d_glm - d_nn)

print(np.percentile(diffs_glm, [2.5, 97.5]))

diffs_gb_nn=[]
for b in range(500):
    idx = rng.integers(0, n, size=n)
    d_gb = mean_poisson_deviance(y_v[idx], lam_gb[idx], sample_weight=e_v[idx])
    d_nn  = mean_poisson_deviance(y_v[idx], lam_nn[idx],  sample_weight=e_v[idx])
    diffs_gb_nn.append(d_gb - d_nn)

print(np.percentile(diffs_gb_nn, [2.5, 97.5]))

calib=pd.DataFrame({})
calib['lam']=lam_gb.copy()
calib['N']=valid['ClaimNb'].to_numpy()
calib['E']=valid['Exposure'].to_numpy()
calib["decile"] = pd.qcut(calib["lam"], 10, labels=False)
calib["pred"] = calib['lam']*calib['E']
table=calib.groupby("decile")[['N','pred']].sum()
table['ratio']=table['N']/table['pred']

print(table)

def calibration(lam):
    calib=pd.DataFrame({})
    calib['lam']=lam.copy()
    calib['N']=valid['ClaimNb'].to_numpy()
    calib['E']=valid['Exposure'].to_numpy()
    calib['decile']=pd.qcut(calib['lam'],10,labels=False)
    calib['pred']=calib['lam']*calib['E']
    table=calib.groupby('decile')[['N','pred']].sum()
    table['ratio']=table['N']/table['pred']
    return(table)

print(calibration(lam_glm))
print(calibration(lam_nn))

facteur_nn=train_sub['ClaimNb'].sum()/(torch.exp(m8(X_train_nn).squeeze(1)).detach().numpy()*train_sub['Exposure']).sum()
lam_nn_corr=lam_nn*facteur_nn
print(facteur_nn)

calibration(lam_nn_corr)['pred'].sum()
print(mean_poisson_deviance(y_valid,lam_nn_corr,sample_weight=valid['Exposure']))


# === Test final ===
y_test = test['ClaimNb'] / test['Exposure']
e_test = test['Exposure']

lam_naif_test = np.full(len(test), f_sub)
lam_glm_test = glm.predict(preparer(test))

X_test_gb = test[colonnes].copy()
X_test_gb["VehGas"] = (X_test_gb["VehGas"] == "'Regular'").astype(float)
lam_gb_test = gb.predict(X_test_gb)

test_norm = preparer3(test, train_sub)
print(list(test_norm.columns) == list(train_sub_norm.columns))
X_test_nn = torch.tensor(test_norm.to_numpy(), dtype=torch.float32)
lam_nn_test = torch.exp(m8(X_test_nn).squeeze(1)).detach().numpy() * facteur_nn

for nom, lam in [("naif", lam_naif_test), ("GLM", lam_glm_test),
                 ("reseau", lam_nn_test), ("gradient boosting", lam_gb_test)]:
    print(nom, mean_poisson_deviance(y_test, lam, sample_weight=e_test))