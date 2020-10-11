import pandas as pd
import numpy as np
df=pd.read_csv('E:/IE517/hm7/ccdefault.csv')
df=df.drop(['ID'],axis=1)
# Split the dataset into a training and a testing set
X = df.drop('DEFAULT',axis=1).values
y = df['DEFAULT'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=30)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Part 1: Random forest estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
CV_scores=[]
for i in [10,30,50,100]:
    rf = RandomForestClassifier(random_state=(42),n_estimators=i)
    rf.fit(X_train,y_train) 
    scores = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10,n_jobs=-1,scoring='accuracy') 
    scores_2 = cross_val_score(estimator=rf, X=X_test, y=y_test, cv=10,n_jobs=-1,scoring='accuracy')  
    CV_scores.append(['cv accuracy in sample'+str(i),np.mean(scores),
                      'cv accuracy out sample'+str(i),np.mean(scores_2)])
CV_scores_table = pd.DataFrame(CV_scores)
print(CV_scores_table)

#Part 2: Random forest feature importance
rf = RandomForestClassifier(random_state=(42),n_estimators=100)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))
feature_names = df.columns[1:]
labels=np.array(feature_names)[sorted_index]
import matplotlib.pyplot as plt
plt.bar(x,importances[sorted_index],tick_label = labels)
plt.xticks(rotation=90)
plt.show()
print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################
