import pandas as pd
import numpy as np
df=pd.read_csv('E:/IE517/hm4/housing.csv')
#arrange data by rows and columns
print(df.shape)
nrow = df.shape[0]
ncol = df.shape[1]
#show the different types in columns
print(df.dtypes)
print(df.head())
#print summary of data frame
summary =df.describe()
print(summary)

#Presenting Attribute Correlations Visually—sampleCorrHeatMap.py
from pandas import DataFrame
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(1)
col = df.columns
cm = np.corrcoef(df[col].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,
               cbar=True,
               annot=True,
               square=True,
               fmt='.2f',
               annot_kws={'size':15},
               yticklabels=(col),
               xticklabels=(col))
plt.show()
#vistualize data with scatter plots 
#using column which has the highest corelation with MEDV
plt.figure(2)
_ = plt.plot(df['MEDV'], df['RM'],marker='.',linestyle='none')
_ = plt.xlabel('MEDV')
_ = plt.ylabel('RM')
plt.show()

X = df.drop('MEDV',axis=1).values
y = df['MEDV'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

#LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)
#coefficient
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('Coefficient:',reg.coef_)
#intercept
print('Intercept:%.3f'%(reg.intercept_))
#residual
plt.figure(3)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolors='white',label='training data')
plt.scatter(y_pred,y_pred-y_test,c='limegreen',marker='s',edgecolors='white',label='test data')
plt.xlabel('predict values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black',lw=2)
plt.xlim([-10,50])
plt.show()
#MSE
from sklearn.metrics import mean_squared_error
print('MSE train:%.3f,test:%.3f'%(mean_squared_error(y_train, y_train_pred),
                                  mean_squared_error(y_test, y_pred)))
#R square
from sklearn.metrics import r2_score
print('R^2 train:%.3f,test:%.3f'%(r2_score(y_train, y_train_pred),
                                  r2_score(y_test, y_pred)))

#RidgeRegression
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1,normalize=(True))
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)
#coefficient
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('Coefficient:',reg.coef_)
#intercept
print('Intercept:%.3f'%(reg.intercept_))
#residual
plt.figure(4)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolors='white',label='training data')
plt.scatter(y_pred,y_pred-y_test,c='limegreen',marker='s',edgecolors='white',label='test data')
plt.xlabel('predict values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black',lw=2)
plt.xlim([-10,50])
plt.show()
#MSE
from sklearn.metrics import mean_squared_error
print('MSE train:%.3f,test:%.3f'%(mean_squared_error(y_train, y_train_pred),
                                  mean_squared_error(y_test, y_pred)))
#R square
from sklearn.metrics import r2_score
print('R^2 train:%.3f,test:%.3f'%(r2_score(y_train, y_train_pred),
                                  r2_score(y_test, y_pred)))

#LassoRegression
from sklearn.linear_model import Lasso
reg = Lasso(alpha=0.1,normalize=(True))
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)
#coefficient
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('Coefficient:',reg.coef_)
#intercept
print('Intercept:%.3f'%(reg.intercept_))
#residual
plt.figure(5)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolors='white',label='training data')
plt.scatter(y_pred,y_pred-y_test,c='limegreen',marker='s',edgecolors='white',label='test data')
plt.xlabel('predict values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black',lw=2)
plt.xlim([-10,50])
plt.show()
#MSE
from sklearn.metrics import mean_squared_error
print('MSE train:%.3f,test:%.3f'%(mean_squared_error(y_train, y_train_pred),
                                  mean_squared_error(y_test, y_pred)))
#R square
from sklearn.metrics import r2_score
print('R^2 train:%.3f,test:%.3f'%(r2_score(y_train, y_train_pred),
                                  r2_score(y_test, y_pred)))
print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################
