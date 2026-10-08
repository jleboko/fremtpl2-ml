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