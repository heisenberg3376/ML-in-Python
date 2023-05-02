import pandas as pd
import quandl # used to get datasets
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

# Here we Predict the Google Stock Closing Price using linear regression
#1
df = quandl.get('WIKI/GOOGL')
# print(df.head()) # gives the entire data with features(Open, High, Low, Close, Volume, Ex-Divided, Split Ratio, Adj.open, Adj.High, Adj.Low, Adj.close, Adj.Volume)
# we only need some features so 
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# open -> opening price of the stock, close -> closing price of the stock
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0 # (high - low) percent
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 # daily percent change

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#2
forecast_col = 'Adj. Close'
df.fillna(value = -99999, inplace=True) # non avaliable data (we cant ignore the *NotaNumber* data so we keep -9999 inplace of na) (treated as an outlier)

forecast_out = int(math.ceil(0.05 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())
#3
X = np.array(df.drop(['label'],axis = 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

#print(forecast_set)

#print(acc)

df['Forecast'] = np.nan

# we create an empty column and fill it with NaN

last_date = df.iloc[-1].name
# used to get the last date
last_unix = last_date.timestamp()
# get the timestamp of it(unix)
one_day = 86400
# add seconds of one day to it
next_unix = last_unix + one_day
# that will be next unix

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    # for every day nextday will be converted form of next unix
    next_unix += 86400
    # (update for nect iteration)
    df.loc[next_date] = [np.nan for k in range(len(df.columns)-1)] + [i]
    # for forecast data, every other column except the last one(forecast) will be NaN

print(acc)
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.show()
