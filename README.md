# Home Price Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("homeprices.csv")
df

dummies=pd.get_dummies(df.town)
dummies


merged=pd.concat([df,dummies],axis='columns')
merged

final=merged.drop(['town','west windsor'],axis='columns')
final


from sklearn.linear_model import LinearRegression
model=LinearRegression()

x=final.drop(['price'],axis='columns')
x

y=final.price
y
model.fit(x,y)
model.predict([[2600,1,0]])
model.predict([[3600,0,1]])
model.score(x,y)
