from __future__ import print_function
import numpy as np
from collections import Counter
from scipy.stats import norm
inf = float('inf')


def bns(features, classes):

    def _true_positive_rate(feature):
        pos_classes = np.where(classes != 0)[0]
        true_positive = np.where(feature[pos_classes] != 0)[0]
        return len(true_positive) / float(len(pos_classes))

    def _false_positive_rate(feature):
        neg_classes = np.where(classes == 0)[0]
        false_positive = np.where(feature[neg_classes] != 0)[0]
        return len(false_positive) / float(len(neg_classes))

    def _bns(feature):
        tpr = _true_positive_rate(feature)
        fpr = _false_positive_rate(feature)
        ratio = np.abs(norm.ppf(tpr) - norm.ppf(fpr))
        return ratio if ratio != inf else 0

    bns_ratio = [_bns(feature) for feature in features]
    return bns_ratio

ouf = open('task.csv')
feat_names = ouf.readline().strip().split(',')
features = []
for line in ouf.readlines():
    features.append(np.array(list(map(float, line.strip().split(',')))))

features = np.array(features)
features = np.transpose(features)

rates = bns(features[1:], features[0])

result = zip(feat_names[1:], rates)
result.sort(key=lambda x: x[1], reverse=True)
print(*result, sep='\n', file=open('bns_results.txt', 'w'))
