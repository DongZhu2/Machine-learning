import pandas as pd
import numpy as np
df=pd.read_csv('E:/IE517/hm6/ccdefault.csv')
df=df.drop(['ID'],axis=1)
#Part 1: Random test train splits
X = df.drop('DEFAULT',axis=1).values
y = df['DEFAULT'].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
steps = [('scaler',StandardScaler()),
         ('tree',DecisionTreeClassifier(criterion='gini',random_state=1))]
pipeline = Pipeline(steps)
accuracy_train=[]
accuracy = []
a_train=[]
a = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=i)
    tree_scaled = pipeline.fit(X_train,y_train)
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy.append(accuracy_score(y_test, y_pred))
    a_train.append([i+1,accuracy_score(y_train, y_train_pred)])
    a.append([i+1,accuracy_score(y_test, y_pred)])
a_train.append(['mean',np.mean(accuracy_train)])
a_train.append(['standard deviation',np.std(accuracy_train)])
a.append(['mean',np.mean(accuracy)])
a.append(['standard deviation',np.std(accuracy)])
accu_train = pd.DataFrame(a_train)
accu = pd.DataFrame(a) 
print(accu_train)
print(accu)  

#Part 2: Cross validation
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)
scores = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=10,n_jobs=-1)  
CV_scores=[]
for i in range(10):
    CV_scores.append([i+1,scores[i]])
CV_scores.append(['mean',np.mean(scores)])
CV_scores.append(['standard deviation',np.std(scores)])
y_pred_cv = pipeline.predict(X_test)
CV_scores.append(['out-of-sample accuracy score',accuracy_score(y_test, y_pred_cv)])
CV_scores_table = pd.DataFrame(CV_scores)
print(CV_scores_table)
print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################