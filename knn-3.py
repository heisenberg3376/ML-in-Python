import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
style.use('fivethirtyeight')
import random


# visit : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# for more info about the dataset

df = pd.read_csv(r'C:\Users\Katta\Downloads\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups')
    distances = []
    for group in data:
        for features in data[group]:
            eucld_dist = np.linalg.norm(np.array(features)- np.array(predict))
            distances.append([eucld_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


full_data = df.astype(float).values.tolist()
# this is to avoid ''s on our numbers
# we shuffle the data before training
random.shuffle(full_data)
# 2 -> benign
# 4 -> malignant

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    train_set [i[-1]].append(i[:-1])

for i in test_data:
    test_set [i[-1]].append(i[:-1])

correct, total = 0,0

for grp in test_set:
    for data in test_set[grp]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if vote == grp:
            correct += 1
        else:
            print(confidence)
        total += 1


print('accuracy :', correct/total)


