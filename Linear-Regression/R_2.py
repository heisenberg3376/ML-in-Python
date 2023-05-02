from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def best_fit_slope_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - (mean(xs*ys))) / (mean(xs)**2 - mean(xs**2)))
    b = (mean(ys) - (m*(mean(xs))))
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def r_sq(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_mean)

def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return (np.array(ys, dtype = np.float64), np.array(xs, dtype = np.float64))
        
xs, ys = create_dataset(40, 80, 2, 'pos')

m, b = best_fit_slope_intercept(xs, ys)

reg_line = [(m*x)+b for x in xs]

rsquared = r_sq(ys, reg_line)
print(rsquared)

predict_x = 8
predict_y = (m*predict_x)+b


##print(m)
##print(b)




plt.scatter(xs, ys)
plt.plot(xs, reg_line)
plt.scatter(predict_x, predict_y, s=100)

plt.show()
