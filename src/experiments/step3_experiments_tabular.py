from os.path import isfile, join
from os import listdir
import time
import json

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score as ps, recall_score as rs, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_dataset(dataset_dir: str, identifier: str, data_type: str) -> list:
    with open(join(dataset_dir, f'{identifier}.graph.{data_type}.tsv')) as file:
        lines = [line.rstrip().split('\t') for line in file]

    return lines


def group_by_relation(lines: list) -> dict:
    grouped = {}
    for line in lines:
        relation = line[1]
        if relation not in grouped:
            grouped[relation] = []
        grouped[relation].append(line)

    return grouped


def build_gt_dict(grouped: dict, entity_id: dict, relation_id: dict) -> dict:
    return {head: int(tail == entity_id['adult_arrest'])
            for head, relation, tail in grouped[relation_id['has_outcome']]
            if tail in [entity_id['invest_cont'], entity_id['adult_arrest']]}


def feature_indexer(feature_label: str,
                    feature_to_entity: dict,
                    relation_id: dict,
                    grouped_train: dict,
                    grouped_valid: dict,
                    grouped_test: dict) -> (dict, dict):
    # collect unique code/category feature
    feature = set()
    if feature_label in relation_id and relation_id[feature_label] in grouped_train:
        feature = feature.union({head if feature_to_entity[feature_label] == 'head' else tail
                                 for head, relation, tail in grouped_train[relation_id[feature_label]]})
    if feature_label in relation_id and  relation_id[feature_label] in grouped_valid:
        feature = feature.union({head if feature_to_entity[feature_label] == 'head' else tail
                                 for head, relation, tail in grouped_valid[relation_id[feature_label]]})
    if feature_label in relation_id and  relation_id[feature_label] in grouped_test:
        feature = feature.union({head if feature_to_entity[feature_label] == 'head' else tail
                                 for head, relation, tail in grouped_test[relation_id[feature_label]]})
    feature_to_pos = {value: idx for idx, value in enumerate(feature)}

    return feature_to_pos, len(feature)


def build_feature_vector(nodes: list,
                         crime_type: str,
                         unique_feature_values: dict,
                         grouped: dict,
                         relation_id: dict,
                         ground_truth: dict) -> (list, np.array):
    x_data, y_data = [], []

    for crime in nodes:
        feature_vectors = {key: [0] * value[1] for key, value in unique_feature_values.items()}

        local_codes = []
        if 'has_category' in feature_vectors and relation_id['has_category'] in grouped:
            for head, relation, tail in grouped[relation_id['has_category']]:
                if head != crime:
                    continue
                local_codes.append(tail)
                feature_vectors['has_category'][unique_feature_values['has_category'][0][tail]] = 1

        if 'has_in_50' in feature_vectors and relation_id['has_in_50'] in grouped:
            for head, relation, tail in grouped[relation_id['has_in_50']]:
                if tail not in local_codes:
                    continue
                feature_vectors['has_in_50'][unique_feature_values['has_in_50'][0][head]] = 1

        local_street = []

        if relation_id[crime_type] in grouped:
            for head, relation, tail in grouped[relation_id[crime_type]]:  # has_CRIMETYPE
                if tail != crime:
                    continue
                local_street.append(head)

            if 'nearby_50_company' in feature_vectors and 'nearby_50_company' in relation_id and relation_id['nearby_50_company'] in grouped:
                for head, relation, tail in grouped[relation_id['nearby_50_company']]:
                    if head not in local_street:
                        continue
                    feature_vectors['nearby_50_company'][unique_feature_values['nearby_50_company'][0][tail]] = 1

            if 'nearby_50_education' in feature_vectors and 'nearby_50_education' in relation_id and relation_id['nearby_50_education'] in grouped:
                for head, relation, tail in grouped[relation_id['nearby_50_education']]:
                    if head not in local_street:
                        continue
                    feature_vectors['nearby_50_education'][unique_feature_values['nearby_50_education'][0][tail]] = 1

            if 'nearby_100_restaurant' in feature_vectors and 'nearby_100_restaurant' in relation_id and relation_id['nearby_100_restaurant'] in grouped:
                for head, relation, tail in grouped[relation_id['nearby_100_restaurant']]:
                    if head not in local_street:
                        continue
                    feature_vectors['nearby_100_restaurant'][unique_feature_values['nearby_100_restaurant'][0][tail]] = 1

            if 'nearby_100_safety' in feature_vectors and 'nearby_100_safety' in relation_id and relation_id['nearby_100_safety'] in grouped:
                for head, relation, tail in grouped[relation_id['nearby_100_safety']]:
                    if head not in local_street:
                        continue
                    feature_vectors['nearby_100_safety'][unique_feature_values['nearby_100_safety'][0][tail]] = 1

            if 'has_poi_in_50' in feature_vectors and 'has_poi_in_50' in relation_id and relation_id['has_poi_in_50'] in grouped:
                for head, relation, tail in grouped[relation_id['has_poi_in_50']]:
                    if head not in local_codes:
                        continue
                    feature_vectors['has_poi_in_50'][unique_feature_values['has_poi_in_50'][0][tail]] = 1

        feature_vector = []
        for _, value in feature_vectors.items():
            feature_vector += value

        x_data.append(np.array(feature_vector))
        y_data.append(ground_truth[crime])

    y_data = np.array(y_data)
    return x_data, y_data


def main():
    # init
    dataset_dir = '/data/crime-knowledge-graph/'
    prefix = 'crime_'
    start_over = False
    bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'

    # load existing files
    only_files = [file for file in listdir(dataset_dir) if isfile(join(dataset_dir, file))]
    identifiers = {file[:file.find('.')] for file in only_files if file.startswith(prefix)}
    identifiers = list(identifiers)
    identifiers.sort()

    # feature set
    feature_sets = {'feature01': ['has_category', 'has_in_50', 'nearby_100_restaurant']}
    feature_to_entity = {'has_category': 'tail',  # crime code
                         'has_in_50': 'head',  # poi identifier
                         'nearby_100_restaurant': 'tail'}

    # remove already processed cases
    file_exists = set()
    for identifier in identifiers:
        for feature_set_id, _ in feature_sets.items():
            output_file = join(dataset_dir, f'{identifier}.tabular.results_{feature_set_id}.json')
            if not start_over and isfile(output_file):
                file_exists.add(output_file)

    for feature_set_id, features in feature_sets.items():
        # run experiments
        for identifier in tqdm(identifiers,
                               total=len(identifiers),
                               desc='Experiments',
                               bar_format=bar_format):
            output_file = join(dataset_dir, f'{identifier}.tabular.results_{feature_set_id}.json')
            if output_file in file_exists:
                continue

            crime_type = None
            if 'burglary' in identifier:
                crime_type = 'has_burglary'
            if 'robbery' in identifier:
                crime_type = 'has_robbery'
            if 'vandalism' in identifier:
                crime_type = 'has_vandalism_-_felony_($400_&_over,_all_church_vandalisms)'

            # load dataset
            train_lines = load_dataset(dataset_dir, identifier, 'train')
            valid_lines = load_dataset(dataset_dir, identifier, 'valid')
            test_lines = load_dataset(dataset_dir, identifier, 'test')

            # load relation to id dictionary
            with open(join(dataset_dir, f'{identifier}.graph.relation2id.txt')) as file:
                lines = [line.rstrip().split('\t') for line in file]
            relation_id = {relation: str(index) for relation, index in lines}

            # load entity to id dictionary
            with open(join(dataset_dir, f'{identifier}.graph.entity2id.txt')) as file:
                lines = [line.rstrip().split('\t') for line in file]
            entity_id = {node: str(index) for node, index in lines}

            # group triples by relation type
            grouped_train = group_by_relation(train_lines)
            grouped_valid = group_by_relation(valid_lines)
            grouped_test = group_by_relation(test_lines)

            # ground truth dictionary
            ground_truth_train = build_gt_dict(grouped_train, entity_id, relation_id)
            ground_truth_valid = build_gt_dict(grouped_valid, entity_id, relation_id)
            ground_truth_test = build_gt_dict(grouped_test, entity_id, relation_id)

            # unique feature values
            unique_feature_values = {
                feature: feature_indexer(feature, feature_to_entity, relation_id,
                                         grouped_train, grouped_valid, grouped_test)
                for feature in features}

            # remove training and validation samples without a valid label
            train_nodes = list({triple[0]
                                for triple in grouped_train[relation_id['has_category']]
                                if triple[0] in ground_truth_train})
            train_nodes.sort()
            valid_nodes = list({triple[0]
                                for triple in grouped_valid[relation_id['has_category']]
                                if triple[0] in ground_truth_valid})
            valid_nodes.sort()
            test_nodes = list({triple[0]
                               for triple in grouped_test[relation_id['has_category']]
                               if triple[0] in ground_truth_test})
            test_nodes.sort()

            # build feature vectors
            x_train, y_train = build_feature_vector(train_nodes, crime_type,
                                                    unique_feature_values, grouped_train,
                                                    relation_id, ground_truth_train)
            x_valid, y_valid = build_feature_vector(valid_nodes, crime_type,
                                                    unique_feature_values, grouped_valid,
                                                    relation_id, ground_truth_valid)
            x_test, y_test = build_feature_vector(test_nodes, crime_type,
                                                  unique_feature_values, grouped_test,
                                                  relation_id, ground_truth_test)

            # randomly picked
            random_states = [(3255946872, 1589303713), (3238745956, 3566716953),
                             (2499942748, 976157507), (760552372, 1145821702),
                             (2368330777, 3490247174), (4126940332, 2557702375),
                             (2682826070, 4161039460), (3674597768, 1161028518),
                             (4158888177, 2652063028), (3899964991, 2038265566)]

            # result dictionary
            results = {'Num Train Samples': str(len(x_train)),
                       'Num Valid Samples': str(len(x_valid)),
                       'Num Test Samples': str(len(x_test)),
                       'Num Positive Train Samples': int(sum(y_train)),
                       'Num Positive Valid Samples': int(sum(y_valid)),
                       'Num Positive Test Samples': int(sum(y_test))}

            # run experiments
            classifiers = ['LogisticRegressionCV', 'RandomForestClassifier']
            for run_idx, (random_a, random_b) in enumerate(random_states):
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
                        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
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
                                       'ROCAUC_Score': float(roc_auc_score(y_test, probabilities[:, 1]))}

                        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
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
