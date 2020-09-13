
import pandas as pd
import numpy as np
data=pd.read_csv('C:/users/Jackt/Desktop/IE517/hm3/HY_Universe_corporate bond.csv')
#delete extreme data of coupon
data = data[-data.Coupon.isin([999])]
#arrange data by rows and columns
print(data.shape)
nrow = data.shape[0]
ncol = data.shape[1]

#show the different types in columns
print(data.dtypes)

#Quantile‐Quantile Plot of Coupon
import pylab
import scipy.stats as stats
stats.probplot(data['Coupon'], dist="norm", plot=pylab)
pylab.show()

#print summary of data frame
summary =data.describe()
print(summary)

#Presenting Attribute Correlations Visually—sampleCorrHeatMap.py
from pandas import DataFrame
corMat = DataFrame(data.corr())
#visualize correlations using heatmap
import matplotlib.pyplot as plt
plt.pcolor(corMat)
plt.show()

#histogram
import matplotlib.pyplot as plt
plt.figure(1)
_ = plt.hist(data['Coupon'])
_ = plt.xlabel('Coupon')
_ = plt.ylabel('number of counties')
plt.show()

#ECDF
plt.figure(2)
x = np.sort(data['Coupon'])
y = np.arange(1,nrow+1)/nrow
_ = plt.plot(x,y,marker='.',linestyle='none')
_ = plt.xlabel('Coupon')
_ = plt.ylabel('ECDF')
plt.margins(0.02)
percentiles=np.array([25,50,75])
ptiles_vers=np.percentile(data['Coupon'], percentiles)
_ = plt.plot(ptiles_vers,percentiles/100,marker='D',color='red',linestyle='none')
plt.show()

#box-plot
plt.figure(3)
import seaborn as sns
_ = sns.boxplot(x='Coupon Type',y='Coupon',data = data)
plt.show()

print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################
