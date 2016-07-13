from __future__ import print_function
import numpy as np
from sklearn.feature_selection import chi2
from collections import Counter
from scipy.stats import norm
inf = float('inf')

ouf = open('task.csv')
feat_names = ouf.readline().strip().split(',')
features = []
for line in ouf.readlines():
    features.append(np.array(list(map(float, line.strip().split(',')))))

features = np.array(features)
features = np.transpose(features)

rates = np.transpose(chi2(np.transpose(features[1:]), features[0])).tolist()

result = sorted(zip(feat_names[1:], rates), key=lambda x: x[1], reverse=True)
print(*result, sep='\n', file=open('chi2_results.txt', 'w'))
