import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Katta\OneDrive\Desktop\canada_per_capita_income.csv')
X = df.drop(['per capita income (US$)'],axis=1)
y = df['per capita income (US$)']
print(df)
print('**************************************************************************')
clf = LinearRegression()
clf.fit(X, y)

pr = pd.DataFrame({'year':[2020,2021,2022,2023,2024]})
print(clf.predict(pr))


plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(X,y,marker='+',s=100,color='r',label='given data')
plt.plot(df['year'],clf.predict(df[['year']]),label='best fit line')
plt.scatter(pr['year'],clf.predict(pr),color='c',marker='*',label='predictions for next 5 years(from 2020)')
plt.legend(loc='lower right')
plt.show()

