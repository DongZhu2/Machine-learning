
#uses Iris database and SGD classifier

from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print( X_iris.shape, y_iris.shape)
#(150, 4) (150,)
print( X_iris[0], y_iris[0])
#(150, 4) (150,)
#[ 5.1  3.5  1.4  0.2] 0

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X, y = X_iris, y_iris
# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=30)
print( X_train.shape, y_train.shape)
#(112, 2) (112,)
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier(criterion='gini',random_state=1)
dt.fit(X_train,y_train)
y_pred_t = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_t)
print(accuracy)

from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
scores.append(accuracy)
print(scores)
print(scores.index(max(scores)))
print('The best choice of K for this data is 4.')
print("My name is Dong Zhu")
print("My NetID is: dongzhu2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################
