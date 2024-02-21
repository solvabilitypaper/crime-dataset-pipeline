import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
import json
from os.path import isfile, join
from os import listdir
import time
from tqdm import tqdm
from sklearn.metrics import precision_score as ps, recall_score as rs, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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

    # adapt the order of the columns
    head = dataset[:, order.find('h')]
    relation = dataset[:, order.find('r')]
    tail = dataset[:, order.find('t')]
    dataset = np.stack((head, relation, tail), axis=1)

    return dataset


def main():
    # init
    dataset_dir = '/data/crime-knowledge-graph/'
    prefix = 'crime_'
    start_over = True
    bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'

    # load existing files
    only_files = [file for file in listdir(dataset_dir) if isfile(join(dataset_dir, file))]
    identifiers = {file[:file.find('.')] for file in only_files if file.startswith(prefix)}
    identifiers = list(identifiers)
    identifiers.sort()

    # identify feature set ids
    feature_set_ids = {}
    for identifier in identifiers:
        prefix_emb = f'{identifier}.graph.node_embeddings'
        feature_set_ids[identifier] = {file[file.rfind('_') + 1:-4]
                                       for file in only_files if file.startswith(prefix_emb)}

    # remove already processed cases
    filtered_identifiers = []
    for identifier in identifiers:
        output_file = join(dataset_dir, f'{identifier}.graph.results.json')
        if not start_over and isfile(output_file):
            print(f'> Skip {identifier} because result file already exists.')
            continue
        filtered_identifiers.append(identifier)

    # run experiments
    for identifier in tqdm(filtered_identifiers,
                           total=len(filtered_identifiers),
                           desc='Experiments',
                           bar_format=bar_format):

        for feature_set_id in feature_set_ids[identifier]:
            output_file = join(dataset_dir, f'{identifier}.graph.results_{feature_set_id}.json')

            with open(join(dataset_dir, f'{identifier}.graph.node_indexer_{feature_set_id}.pkl'), 'rb') as handle:
                node_indexer = pickle.load(handle)

            with open(join(dataset_dir, f'{identifier}.graph.node_embeddings_{feature_set_id}.pkl'), 'rb') as handle:
                node_embeddings = pickle.load(handle)

            emb_index = []
            for idx, indexer in node_indexer.items():
                emb_index.append((node_embeddings[idx], indexer))

            # load dataset
            train_data = load_dataset(join(dataset_dir, f'{identifier}.graph.train.tsv'), delimiter='\t')
            valid_data = load_dataset(join(dataset_dir, f'{identifier}.graph.valid.tsv'), delimiter='\t')
            test_data = load_dataset(join(dataset_dir, f'{identifier}.graph.test.tsv'), delimiter='\t')

            # load relation to id dictionary
            with open(join(dataset_dir, f'{identifier}.graph.relation2id.txt')) as file:
                lines = [line.rstrip().split('\t') for line in file]
            relation_id = {relation: str(index) for relation, index in lines}

            # load entity to id dictionary
            with open(join(dataset_dir, f'{identifier}.graph.entity2id.txt')) as file:
                lines = [line.rstrip().split('\t') for line in file]
            entity_id = {node: str(index) for node, index in lines}

            if 'invest_cont' not in entity_id or 'adult_arrest' not in entity_id:
                with open(output_file, 'w', encoding='utf-8') as writer:
                    writer.write(json.dumps({}, indent=4) + '\n')
                continue

            # ground truth dictionary
            train_crime_label = {row[0]: row[2] for row in train_data
                                 if str(row[1]) == relation_id['has_outcome']
                                 and str(row[2]) in [entity_id['invest_cont'], entity_id['adult_arrest']]}
            valid_crime_label = {row[0]: row[2] for row in valid_data
                                 if str(row[1]) == relation_id['has_outcome']
                                 and str(row[2]) in [entity_id['invest_cont'], entity_id['adult_arrest']]}
            test_crime_label = {row[0]: row[2] for row in test_data
                                if str(row[1]) == relation_id['has_outcome']
                                and str(row[2]) in [entity_id['invest_cont'], entity_id['adult_arrest']]}

            # remove training and validation samples without a valid label
            train_nodes = list({row[0]
                                for row in train_data
                                if str(row[1]) == relation_id['has_category'] and row[0] in train_crime_label})
            train_nodes.sort()
            valid_nodes = list({row[0]
                                for row in valid_data
                                if str(row[1]) == relation_id['has_category'] and row[0] in valid_crime_label})
            valid_nodes.sort()
            test_nodes = list({row[0]
                               for row in test_data
                               if str(row[1]) == relation_id['has_category'] and row[0] in test_crime_label})
            test_nodes.sort()

            # randomly picked: randrange(2**32-1)
            # Note: The number of states (i.e., runs) has to match the number of
            # states/runs in step2
            random_states = [(3255946872, 1589303713), (3238745956, 3566716953),
                             (2499942748, 976157507), (760552372, 1145821702),
                             (2368330777, 3490247174), (4126940332, 2557702375),
                             (2682826070, 4161039460), (3674597768, 1161028518),
                             (4158888177, 2652063028), (3899964991, 2038265566)]

            # result dictionary
            results = {}

            # run experiments
            classifiers = ['LogisticRegressionCV', 'RandomForestClassifier']
            for run_idx, (random_a, random_b) in enumerate(random_states):
                tmp_node_embeddings, tmp_node_indexer = emb_index[run_idx]

                # build training and validation data structure for classifier
                x_train = np.array([tmp_node_embeddings[tmp_node_indexer[entry]] for entry in train_nodes])
                y_train = np.array(
                    [int(str(train_crime_label[entry]) == entity_id['adult_arrest']) for entry in train_nodes])
                x_valid = np.array([tmp_node_embeddings[tmp_node_indexer[entry]] for entry in valid_nodes])
                y_valid = np.array(
                    [int(str(valid_crime_label[entry]) == entity_id['adult_arrest']) for entry in valid_nodes])
                x_test = np.array([tmp_node_embeddings[tmp_node_indexer[entry]] for entry in test_nodes])
                y_test = np.array([int(str(test_crime_label[entry]) == entity_id['adult_arrest']) for entry in test_nodes])

                if 'Num Train Samples' not in results:
                    results['Num Train Samples'] = str(x_train.shape)
                    results['Num Valid Samples'] = str(x_valid.shape)
                    results['Num Test Samples'] = str(x_test.shape)
                    results['Num Positive Train Samples'] = int(sum(y_train))
                    results['Num Positive Valid Samples'] = int(sum(y_valid))
                    results['Num Positive Test Samples'] = int(sum(y_test))

                for classifier in classifiers:
                    if classifier not in results:
                        results[classifier] = {}
                    results[classifier][f'Run_{run_idx}'] = {}

                    try:
                        # train ML model
                        folds = 10
                        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_a)
                        if classifier == 'LogisticRegressionCV':
                            clf = LogisticRegressionCV(cv=skf, random_state=random_b, solver='liblinear')
                        elif classifier == 'RandomForestClassifier':
                            rfc = RandomForestClassifier(random_state=random_b)
                            param_grid = {
                                'n_estimators': [5, 10, 50, 100, 200],
                                'max_depth': [5, 10, 25, 50]
                            }
                            clf = GridSearchCV(rfc, param_grid, cv=skf)
                        else:
                            raise ValueError('Invalid Classifier!')
                        clf.fit(x_train, y_train)

                        # determine threshold
                        probabilities_valid = clf.predict_proba(x_valid)
                        best_value = -1
                        best_threshold = -1
                        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                          0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
                            predictions = [1 if value > threshold else 0 for value in probabilities_valid[:, 1]]
                            f1_score_class_1 = float(f1_score(y_valid, predictions, pos_label=1))
                            if f1_score_class_1 > best_value:
                                best_value = f1_score_class_1
                                best_threshold = threshold

                        # run predictions
                        probabilities = clf.predict_proba(x_test)

                        # save result
                        run_results = {'CV': folds,
                                       'Random_State': (random_a, random_b),
                                       'Threshold': best_threshold,
                                       'Valid F1 (Class 1)': best_value,
                                       'predict_proba': probabilities_valid,
                                       'ROCAUC_Score': float(roc_auc_score(y_test, probabilities[:, 1]))}

                        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                          0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
                            predictions = [1 if value > threshold else 0 for value in probabilities[:, 1]]
                            threshold_results = {'PS_Macro': float(ps(y_test, predictions, average='macro')),
                                                 'PS_Micro': float(ps(y_test, predictions, average='micro')),
                                                 'PS_Weighted': float(ps(y_test, predictions, average='weighted')),
                                                 'Precision (Class 0)': float(
                                                     ps(y_test, predictions, average='binary', pos_label=0)),
                                                 'Recall (Class 0)': float(
                                                     rs(y_test, predictions, average='binary', pos_label=0)),
                                                 'F1 (Class 0)': float(f1_score(y_test, predictions, pos_label=0)),
                                                 'F1 (Class 1)': float(f1_score(y_test, predictions, pos_label=1)),
                                                 'B. Accuracy': float(balanced_accuracy_score(y_test, predictions)),
                                                 'CM': str(confusion_matrix(y_test, predictions)).replace('\n ', '')}
                            run_results[f'Threshold_{threshold}'] = threshold_results
                        results[classifier][f'Run_{run_idx}'] = run_results
                    except ValueError:
                        pass  # probably not enough training data

            with open(output_file, 'w', encoding='utf-8') as writer:
                writer.write(json.dumps(results, indent=4) + '\n')


if __name__ == "__main__":
    start = time.time()
    main()
    print(f'Runtime: {time.time() - start}')
