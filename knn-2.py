import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

# we create a model dataset to build our own KNN classifier
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_feat = [5,5]

##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0], ii[1], s=100, color=i)

##eucld_dist = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)
##print(eucld_dist)

# the no. of groups must always be lesser than the no. of neighbors(k value)

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups')
        
    distances = []
    for group in data:
        for features in data[group]:
            eucld_dist = np.linalg.norm(np.array(features)- np.array(predict))
            # to find the euclidean distance 
            distances.append([eucld_dist, group])
            # distances will be a list of list holding the euclidean dist, group('k' or 'r')

    votes = [i[1] for i in sorted(distances)[:k]]
    # we only care about the smallest, first k values
    vote_result = Counter(votes).most_common(1)[0][0]
    # whatever is the most common in those values will be the vote result
    return vote_result

res = k_nearest_neighbors(dataset, new_feat, k=3)
print(res)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feat[0],new_feat[1] ,s=100, color = res)
plt.show()
