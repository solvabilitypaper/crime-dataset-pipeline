import sys

sys.path.append('/root/crime-knowledge-graph/src')

import os
import math
import pickle
import time
from tqdm import tqdm

from api.database import DatabaseAPI


def get_graph(crime_type: str, district: str) -> (dict, list):
    time_to_graph = {}
    poi_types = ['safety', 'government', 'company', 'restaurant', 'education']

    # min: 1262340000, max: 1665427800, One week in seconds: 604800
    num_triples = []
    bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
    start_time = 1262340000
    end_time = 1665427800
    step = 604800
    total = math.ceil((end_time - start_time) / step)
    for idx, value in tqdm(enumerate(range(start_time, end_time, step)),
                           total=total,
                           desc='[Build Snapshots',
                           bar_format=bar_format):
        start_time = value
        end_time = value + 604800
        triples, nodes = build(start_time, end_time, district, crime_type, poi_types)

        time_to_graph[idx] = [triples, nodes]
        num_triples.append(len(triples))

    return time_to_graph, num_triples


def build(start_time: int,
          end_time: int,
          district: str,
          crime_type: str,
          poi_types: list) -> (list, set):
    # establish database connection
    db = DatabaseAPI()

    # load district data
    district_id = db.select('district',
                            {'source': 'geo_los_angeles',
                             'properties.name': district},
                            {'_id': 1})

    # load crime data
    crime_selector = {'location.neighbourhood': district_id[0]['_id'],
                      'timestamp.occurred': {'$gt': start_time,
                                             '$lt': end_time},
                      'crime_type.label': crime_type}
    crime_filter = {'_id': 1, 'sid': 1, 'crime_type.label': 1,
                    'nearby': 1, 'case.description': 1, 'case.label': 1,
                    'streets': 1}
    crime_data = db.select('crime', crime_selector, crime_filter)
    crime_data = {str(entry['_id']): entry for entry in crime_data}

    # load poi data
    poi_selector = {'location.neighbourhood': district_id[0]['_id'],
                    'category': {'$in': poi_types}}
    poi_filter = {'_id': 1, 'sid': 1, 'category': 1, 'properties': 1}
    poi_data = db.select('poi', poi_selector, poi_filter)
    poi_data = {str(entry['_id']): entry for entry in poi_data}

    # load street data
    street_ids = set()
    for crime_id, values in crime_data.items():
        street_ids = street_ids.union(set(values['streets']))

    street_selector = {'_id': {'$in': list(street_ids)}}
    street_filter = {'_id': 1, 'sid': 1, 'location.street': 1,
                     'crimes': 1, 'nearby': 1, 'location.neighbourhood': 1}
    street_data = db.select('street', street_selector, street_filter)

    # dataset
    node_set = set()
    triple_list = []

    # build triples
    for entry in street_data:
        street_name = str(entry['sid'])
        num_triples = len(triple_list)

        # street-to-poi
        for radii, pois in entry['nearby'].items():
            for poi_key in pois:
                poi_key = str(poi_key)
                if poi_key not in poi_data:
                    continue

                poi_name = str(poi_data[poi_key]['sid'])
                poi_category = poi_data[poi_key]['category']
                triple_list.append((street_name,
                                    f'nearby_{radii}_{poi_category}',
                                    poi_name))
                node_set.add(poi_name)

        # street-to-crime & crime-to-crime_code
        for crime_key in entry['crimes']:
            crime_key = str(crime_key)
            if crime_key not in crime_data:
                continue

            crime_outcome = crime_data[crime_key]['case']['label']
            crime_category = crime_data[crime_key]['crime_type']['label']
            crime_description = crime_data[crime_key]['case']['description']

            triple_list.append((street_name,
                                f'has_{crime_category}',
                                crime_key))
            triple_list.append((crime_key, f'has_outcome', crime_outcome))

            for crime_code in crime_description:
                crime_code = str(crime_code)
                triple_list.append((crime_key, f'has_category', crime_code))
                node_set.add(crime_code)

        # add street node if triples were generated
        if len(triple_list) > num_triples:
            node_set.add(street_name)

    # crime_code-to-poi
    # crime-to-poi
    for crime_key, crimes in crime_data.items():
        if 'nearby' not in crimes:
            continue

        crime_description = crimes['case']['description']
        crime_nearby = crimes['nearby']

        for crime_code in crime_description:
            for nearby_value in ['50', '100']:  # meters
                if nearby_value not in crime_nearby:
                    continue

                for poi_key in crime_nearby[nearby_value]:
                    if str(poi_key) not in poi_data:
                        continue
                    poi_name = str(poi_data[str(poi_key)]['sid'])
                    triple_list.append((poi_name,
                                        f'has_in_{nearby_value}',
                                        str(crime_code)))
                    node_set.add(poi_name)

                    triple_list.append((str(crime_key),
                                        f'has_poi_in_{nearby_value}',
                                        poi_name))
            node_set.add(str(crime_code))

    triple_list = list(set(triple_list))
    triple_list.sort()

    return triple_list, node_set


def save_graph(time_to_graph: dict, output_dir: str):
    with open(output_dir, 'wb') as file:
        pickle.dump(time_to_graph, file)


def load_graph(output_dir: str) -> dict:
    with open(output_dir, 'rb') as file:
        loaded_dict = pickle.load(file)

    return loaded_dict


def create_dataset(time_to_graph: dict) -> (tuple, tuple):
    nodes_set = set()

    tkg_train = []
    tkg_valid = []
    tkg_test = []

    node_indexer = {}
    relation_indexer = {}

    for timestep, (triples, nodes) in time_to_graph.items():
        for triple in triples:
            head = triple[0].replace(' ', '_')
            if head not in node_indexer:
                node_indexer[head] = len(node_indexer)
            head_idx = node_indexer[head]

            tail = triple[2].replace(' ', '_')
            if tail not in node_indexer:
                node_indexer[tail] = len(node_indexer)
            tail_idx = node_indexer[tail]

            relation = triple[1].replace(' ', '_')
            if triple[1] not in relation_indexer:
                relation_indexer[relation] = len(relation_indexer)
            relation_idx = relation_indexer[relation]

            if timestep < 532:
                tkg_train.append([head_idx, relation_idx, tail_idx, timestep])
            if 532 <= timestep < 600:
                tkg_valid.append([head_idx, relation_idx, tail_idx, timestep])
            if timestep > 600:
                tkg_test.append([head_idx, relation_idx, tail_idx, timestep])

        nodes_set = nodes_set.union(nodes)

    print(f'Number of Nodes: {len(nodes_set)}')

    return (tkg_train, tkg_valid, tkg_test), (node_indexer, relation_indexer)


def save_dataset(crime_type_label: str, district_label: str, dataset: tuple, indexer: tuple, output_dir: str):
    tkg_train, tkg_valid, tkg_test = dataset
    node_indexer, relation_indexer = indexer

    prefix = f'crime_{crime_type_label}_{district_label}'.lower().replace(' ', '_')
    dataset_files = {'.graph.train.tsv': tkg_train,
                     '.graph.valid.tsv': tkg_valid,
                     '.graph.test.tsv': tkg_test}
    indexer_files = {'.graph.entity2id.txt': node_indexer,
                     '.graph.relation2id.txt': relation_indexer}

    for filename, dataset in dataset_files.items():
        file_path = os.path.join(output_dir, prefix + filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in dataset:
                file.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')
                # file.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t'+str(line[3]) + '\n')

    for filename, indexer in indexer_files.items():
        file_path = os.path.join(output_dir, prefix + filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            for key, value in indexer.items():
                label = str(key).replace('\n', '').replace('\t', '').replace('\r', '').rstrip()
                file.write(label + '\t' + str(value) + '\n')


def get_label(value: str) -> str:
    return value.replace(' ', '-').replace('_', '-').replace(',', '-').replace('/', '-').replace('.', '-')


def main():
    output_dir = '/data/crime-knowledge-graph/'
    runtime = time.time()
    print('Load data from database ...')

    datasets = {'burglary': ['San Pedro', 'Koreatown', 'Van Nuys', 'Woodland Hills', 'Downtown'],
                'robbery': ['Florence', 'Koreatown', 'Hollywood', 'Westlake', 'Downtown'],
                'vandalism - felony ($400 & over, all church vandalisms)': ['Boyle Heights', 'Westlake', 'Van Nuys', 'Hollywood', 'Downtown']}

    for crime_type, districts in datasets.items():
        for district in districts:
            # load data from the database
            time_to_graph, num_triples = get_graph(crime_type, district)

            # save graph snapshot
            crime_type_label = get_label(crime_type)
            district_label = get_label(district)
            file_path = f'crime_{crime_type_label}_{district_label}.graph.pkl'.lower().replace(' ', '_')
            file_path = os.path.join(output_dir, file_path)
            save_graph(time_to_graph, file_path)
            time_to_graph = load_graph(file_path)

            # print some stats
            print(f'> Time steps: {len(time_to_graph)}')
            print(f'> Triples: {min(num_triples)} (min) | '
                  f'{max(num_triples)} (max) | '
                  f'{sum(num_triples) / len(num_triples)} (avg)')
            print(f'> Done ({"{:.2f}".format(time.time() - runtime)} seconds)')

            # split graph into train, valid, and test set
            dataset, indexer = create_dataset(time_to_graph)

            # save dataset to disk as tsv files
            save_dataset(crime_type_label, district_label, dataset, indexer, output_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f'Runtime: {time.time()-start}')
