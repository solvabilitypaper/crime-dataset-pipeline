"""
    crime-dataset-pipeline

       File: tabular.py

    Authors: Deleted for purposes of anonymity

    Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION

The software and its source code contain valuable trade secrets and shall be
maintained in confidence and treated as confidential information. The software
may only be used for evaluation and/or testing purposes, unless otherwise
explicitly stated in the terms of a license agreement or nondisclosure
agreement with the proprietor of the software. Any unauthorized publication,
transfer to third parties, or duplication of the object or source
code---either totally or in part---is strictly prohibited.

    Copyright (c) 2023 Proprietor: Deleted for purposes of anonymity
    All Rights Reserved.

THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.

NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES.

For purposes of anonymity, the identity of the proprietor is not given
herewith. The identity of the proprietor will be given once the review of the
conference submission is completed.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import sklearn

filename_train = 'crime_burglary.train.txt'
with open(filename_train) as file:
    train_lines = [line.rstrip().split() for line in file]

filename_test = 'crime_burglary.test.txt'
with open(filename_test) as file:
    test_lines = [line.rstrip().split() for line in file]

grouped_train = {}
for line in train_lines:
    relation = line[1]
    if relation not in grouped_train:
        grouped_train[relation] = []
    grouped_train[relation].append(line)

grouped_test = {}
for line in test_lines:
    relation = line[1]
    if relation not in grouped_test:
        grouped_test[relation] = []
    grouped_test[relation].append(line)

ground_truth_train = {head: 0 if tail == '1488' else 1 for head, relation, tail in grouped_train['28'] if
                      tail in ['1488', '1814']}
ground_truth_test = {head: 0 if tail == '1488' else 1 for head, relation, tail in grouped_test['28'] if
                     tail in ['1488', '1814']}

codes = {tail for head, relation, tail in grouped_train['24']}  # [str(crime_key), f'has_category', str(crime_code)]
codes = codes.union({tail for head, relation, tail in grouped_test['24']})
code_to_pos = {value: idx for idx, value in enumerate(codes)}
num_codes = len(codes)

pois = {head for head, relation, tail in grouped_train['30']}  # [poi_name, f'has_in_50', str(crime_code)]
pois = pois.union({head for head, relation, tail in grouped_test['30']})
pois_to_pos = {value: idx for idx, value in enumerate(pois)}
num_pois = len(pois)

buglary_nodes_train = list(ground_truth_train.keys())
buglary_nodes_train.sort()

buglary_nodes_test = list(ground_truth_test.keys())
buglary_nodes_test.sort()

x_train = []
y_train = []

for burglary in tqdm(buglary_nodes_train):
    code_features = [0] * num_codes
    poi_features = [0] * num_pois

    local_codes = []
    for head, relation, tail in grouped_train['24']:
        if head != burglary:
            continue
        local_codes.append(tail)
        code_features[code_to_pos[tail]] = 1

    for head, relation, tail in grouped_train['30']:
        if tail not in local_codes:
            continue
        poi_features[pois_to_pos[head]] = 1

    x_train.append(np.array(code_features + poi_features))
    y_train.append(ground_truth_train[burglary])

y_train = np.array(y_train)

x_test = []
y_test = []

for burglary in tqdm(buglary_nodes_test):
    code_features = [0] * num_codes
    poi_features = [0] * num_pois

    local_codes = []
    for head, relation, tail in grouped_test['24']:
        if head != burglary:
            continue
        local_codes.append(tail)
        code_features[code_to_pos[tail]] = 1

    for head, relation, tail in grouped_test['30']:
        if tail not in local_codes:
            continue
        poi_features[pois_to_pos[head]] = 1

    x_test.append(np.array(code_features + poi_features))
    y_test.append(ground_truth_test[burglary])

clf = LogisticRegressionCV(cv=10, random_state=0, solver='liblinear').fit(x_train, y_train)
predictions2 = clf.predict_proba(x_test)
print(roc_auc_score(y_test, predictions2[:, 1]))

print('roc_auc', sklearn.metrics.roc_auc_score(y_test, predictions2[:, 1]))

for value2 in range(10):
    print(f'Threshold: {value2}')
    predictions = [1 if value > (value2 / 10) else 0 for value in predictions2[:, 1]]
    print(confusion_matrix(y_test, predictions))
    a = precision_score(y_test, predictions, average='macro')
    print(a)
    b = precision_score(y_test, predictions, average='micro')
    print(b)
    c = precision_score(y_test, predictions, average='weighted')
    print(c)
    print('F1 score 0', sklearn.metrics.f1_score(y_test, predictions, pos_label=0))
    print('F1 score 1', sklearn.metrics.f1_score(y_test, predictions, pos_label=1))

    # print('F1 score', sklearn.metrics.f1_score(y_test, predictions))
    # print('F1 score', sklearn.metrics.f1_score(y_test, predictions, average='weighted'))
    # print('Balanced Accuracy score', sklearn.metrics.balanced_accuracy_score(y_test, predictions))
