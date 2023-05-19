import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = 4*np.random.rand(200,1) -2
y = 5*x**2 + 2*x + 17 + 3*np.random.randn(200,1)
poly = PolynomialFeatures(degree=8)
x_poly = poly.fit_transform(x)
x_vals = np.linspace(-2,2,100).reshape(-1,1)
x_vals_poly = poly.transform(x_vals)
clf = LinearRegression()
clf.fit(x_poly,y)
plt.scatter(x,y)
plt.plot(x_vals,clf.predict(x_vals_poly))
plt.show()
#print(y)
#print(x_vals)
