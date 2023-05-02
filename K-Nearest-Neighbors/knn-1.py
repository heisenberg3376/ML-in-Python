import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

# visit : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# for more info about the dataset

df = pd.read_csv(r'C:\Users\Katta\Downloads\breast-cancer-wisconsin.data')
# the data has some question marks so we fill them with an outlier
df.replace('?', -99999, inplace=True)
# we drop the id column as it does not have any impact on the class 
df.drop(['id'], axis=1, inplace=True)

# X will be all the columns, except the class
# y will be the class column
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(acc)
#print(df.head())


exmp = np.array([[4,2,1,1,1,2,3,2,1],[6,4,1,8,0,2,3,2,1]])

exmp = exmp.reshape(len(exmp),-1)
pred = clf.predict(exmp)

print(pred)
