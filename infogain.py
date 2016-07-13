from __future__ import print_function
import numpy as np
from collections import Counter


def infogain(features, classes):

    def _entropy(x, y):
        a = x / float(x + y)
        b = y / float(x + y)
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return b * np.log2(b)
        if b == 0:
            return a * np.log2(a)
        return -(a * np.log2(a) + b * np.log2(b))


    def _infogain(feature):
        true_positive = np.where(feature[pos_classes] != 0)[0]
        false_positive = np.where(feature[neg_classes] != 0)[0]
        p_word = (len(true_positive) + len(false_positive)) / float((len(pos_classes) + len(neg_classes)))
        return start_entr - (p_word * _entropy(len(true_positive), len(false_positive)) +
                             (1 - p_word) * _entropy(len(pos_classes) - len(true_positive),
                                                    len(neg_classes) - len(false_positive)))

    pos_classes = np.where(classes != 0)[0]
    neg_classes = np.where(classes == 0)[0]
    start_entr = _entropy(len(pos_classes), len(neg_classes))
    infogain_ratio = [_infogain(feature) for feature in features]
    return infogain_ratio

ouf = open('task.csv')
feat_names = ouf.readline().strip().split(',')
features = []
for line in ouf.readlines():
    features.append(np.array(list(map(float, line.strip().split(',')))))

features = np.array(features)
features = np.transpose(features)

rates = infogain(features[1:], features[0])

result = zip(feat_names[1:], rates)
result.sort(key =lambda x: x[1], reverse=True)
print(*result, sep='\n', file=open('infogain_results_new.txt', 'w'))