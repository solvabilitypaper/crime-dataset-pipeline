import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
import sklearn
from sklearn.metrics import precision_score, recall_score


def load_dataset(file_path: str,
                 has_headline: bool = False,
                 delimiter: str = ' ',
                 comments: str = '#',
                 order: str = 'hrt') -> np.array or None:

    # parse data to numpy array
    dataset = np.genfromtxt(file_path,
                            comments=comments,
                            delimiter=delimiter,
                            dtype=str)

    # remove potential headline
    if has_headline:
        dataset = dataset[1:]

    # numerical dtypes
    num_dtypes = [int, float]

    # adapt the order of the columns
    head = dataset[:, order.find('h')]
    relation = dataset[:, order.find('r')]
    tail = dataset[:, order.find('t')]
    dataset = np.stack((head, relation, tail), axis=1)

    return dataset


with open('vertex_indexer.pickle', 'rb') as handle:
    vertex_indexer = pickle.load(handle)

with open('weights.pickle', 'rb') as handle:
    weights = pickle.load(handle)

train_data = load_dataset('crime_burglary.train.txt', delimiter='\t')
valid_data = load_dataset('crime_burglary.valid.txt', delimiter='\t')
test_data = load_dataset('crime_burglary.test.txt', delimiter='\t')

train_crime_label = {row[0]: row[2] for row in train_data if row[1] == '28' and row[2] in ['1488', '1814'] and row[0]}
# train_nodes = list(train_crime_label.keys())
train_nodes = list({row[0] for row in train_data if row[1] == '24' and row[0] in train_crime_label})
train_nodes.sort()
x_train = np.array([weights[vertex_indexer[entry]] for entry in train_nodes])
y_train = np.array([1 if int(train_crime_label[entry])==1814 else 0 for entry in train_nodes])

print(x_train.shape)
print(y_train.shape)
print(sum(y_train))

valid_crime_label = {row[0]: row[2] for row in valid_data if row[1] == '28' and row[2] in ['1488', '1814']}
valid_nodes = list(valid_crime_label.keys())
valid_nodes.sort()
x_valid = np.array([weights[vertex_indexer[entry]] for entry in valid_nodes])
y_valid = np.array([1 if int(valid_crime_label[entry])==1814 else 0 for entry in valid_nodes])

print(x_valid.shape)
print(y_valid.shape)
print(sum(y_valid))

test_crime_label = {row[0]: row[2] for row in test_data if row[1] == '28' and row[2] in ['1488', '1814']}
test_nodes = list(test_crime_label.keys())
test_nodes.sort()
x_test = np.array([weights[vertex_indexer[entry]] for entry in test_nodes])
y_test = np.array([1 if int(test_crime_label[entry])==1814 else 0 for entry in test_nodes])

print(x_test.shape)
print(y_test.shape)
print(sum(y_test))

clf = LogisticRegressionCV(cv=10, random_state=0, solver='liblinear').fit(x_train, y_train)
predictions2 = clf.predict_proba(x_test)
print(roc_auc_score(y_test, predictions2[:, 1]))

for value2 in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    print(f'Threshold: {value2}')
    predictions = [1 if value > value2 else 0 for value in predictions2[:, 1]]
    print(confusion_matrix(y_test, predictions))
    a = precision_score(y_test, predictions, average='macro')
    print(a)
    b = precision_score(y_test, predictions, average='micro')
    print(b)
    c = precision_score(y_test, predictions, average='weighted')
    print(c)

    d = precision_score(y_test, predictions, average='binary', pos_label=0)
    print('Precision Score', d)
    e = recall_score(y_test, predictions, average='binary', pos_label=0)
    print('Recall Score', e)

    print('F1 score 0', sklearn.metrics.f1_score(y_test, predictions, pos_label=0))
    print('F1 score 1', sklearn.metrics.f1_score(y_test, predictions, pos_label=1))
    # print('F1 score', sklearn.metrics.f1_score(y_test, predictions, average='weighted'))
    print('Balanced Accuracy score', sklearn.metrics.balanced_accuracy_score(y_test, predictions))
