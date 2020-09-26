import pandas as pd
import numpy as np
df=pd.read_csv('E:/IE517/hm5/hw5_treasury yield curve data.csv')
df=df.drop(['Date'],axis=1)
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
               annot=False,
               square=True,
               fmt='.2f',
               annot_kws={'size':15},
               yticklabels=(col),
               xticklabels=(col))
plt.show()
#vistualize data with box-plot
plt.figure(2)
import seaborn as sns
_ = sns.boxplot(y = df['Adj_Close'])
plt.show()

X = df.drop('Adj_Close',axis=1).values
y = df['Adj_Close'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15, random_state=42)

#Compute and display the explained variance ratio for all components
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)
#Compute and display the explained variance ratio on n_components=3
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

#Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#Original
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
#PCA
X_test_pca = pca.transform(X_test)
reg.fit(X_train_pca,y_train)
y_pred_pca = reg.predict(X_test_pca)
#RMSE
from sklearn.metrics import mean_squared_error
a = mean_squared_error(y_test, y_pred)
b = mean_squared_error(y_test, y_pred_pca)
print('RMSE original:%.3f,pca:%.3f'%(a**0.5,b**0.5))
#R square
from sklearn.metrics import r2_score
print('R^2 original:%.3f,pca:%.3f'%(r2_score(y_test, y_pred),
                                  r2_score(y_test, y_pred_pca)))

#SVM
from sklearn.svm import SVR
svm = SVR()
#Original
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
#PCA
X_test_pca = pca.transform(X_test)
svm.fit(X_train_pca,y_train)
y_pred_pca = svm.predict(X_test_pca)
#RMSE
from sklearn.metrics import mean_squared_error
a = mean_squared_error(y_test, y_pred)
b = mean_squared_error(y_test, y_pred_pca)
print('RMSE original:%.3f,pca:%.3f'%(a**0.5,b**0.5))
#R square
from sklearn.metrics import r2_score
print('R^2 original:%.3f,pca:%.3f'%(r2_score(y_test, y_pred),
                                  r2_score(y_test, y_pred_pca)))

print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################