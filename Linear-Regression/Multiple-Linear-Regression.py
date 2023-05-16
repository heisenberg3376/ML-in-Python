import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\Katta\Downloads\job.csv')
exp = {np.nan:0,'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11}
#df.replace({'experience':exp},inplace=True)
df['experience'] = df['experience'].map(exp)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace=True)

X = df.drop(['salary($)'],axis=1)
y = df['salary($)']

clf = LinearRegression()
clf.fit(X.values,y)
preds = clf.predict([[2,9.0,6]])
print(preds)


