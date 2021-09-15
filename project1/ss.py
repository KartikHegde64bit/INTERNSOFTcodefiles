import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

data=pd.read_csv("advertising.csv")
print(data.head())
fig , axs = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])
plt.show()

feature=['TV']
x=data[feature]
y=data.Sales
lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(x_new.head())

preds=lr.predict(x_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=1)

lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()
lm.rsquared

features=['TV','Radio','Newspaper']
x=data[features]
y=data.Sales

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()


