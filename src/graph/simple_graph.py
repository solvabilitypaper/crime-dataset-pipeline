"""
    crime-dataset-pipeline

       File: simple_graph.py

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

from tqdm import tqdm

from api.database import DatabaseAPI


def get_snapshot(start: int,
                 end: int,
                 district: str,
                 crime_types: list,
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
                      'timestamp.occurred': {'$gt': start,
                                             '$lt': end},
                      'crime_type.label': {'$in': crime_types}}
    crime_filter = {'_id': 1, 'sid': 1, 'crime_type.label': 1,
                    'nearby': 1, 'case.description': 1, 'case.label': 1}
    crime_data = db.select('crime', crime_selector, crime_filter)
    crime_data = {str(entry['_id']): entry for entry in crime_data}

    # load poi data
    poi_selector = {'location.neighbourhood': district_id[0]['_id'],
                    'category': {'$in': poi_types}}
    poi_filter = {'_id': 1, 'sid': 1, 'category': 1, 'properties': 1}
    poi_data = db.select('poi', poi_selector, poi_filter)
    poi_data = {str(entry['_id']): entry for entry in poi_data}

    # load street data
    street_selector = {'location.neighbourhood': district_id[0]['_id']}
    street_filter = {'_id': 1, 'sid': 1, 'location.street': 1,
                     'crimes': 1, 'nearby': 1}
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
                triple_list.append([street_name, f'nearby_{radii}_{poi_category}', poi_name])
                node_set.add(poi_name)

        # street-to-crime & crime-to-crime_code
        for crime_key in entry['crimes']:
            crime_key = str(crime_key)
            if crime_key not in crime_data:
                continue

            crime_outcome = crime_data[crime_key]['case']['label']
            crime_category = crime_data[crime_key]['crime_type']['label']
            crime_description = crime_data[crime_key]['case']['description']

            triple_list.append([street_name, f'has_{crime_category}', str(crime_key)])
            triple_list.append([str(crime_key), f'has_outcome', crime_outcome])

            for crime_code in crime_description:
                triple_list.append([str(crime_key), f'has_category', str(crime_code)])
                node_set.add(str(crime_code))

        # add street node if triples were generated
        if len(triple_list) > num_triples:
            node_set.add(street_name)

    # crime_code-to-poi
    # crime-to-poi
    for crime_key, crimes in crime_data.items():
        if 'nearby' not in crimes:
            continue

        # crime_category = crimes['crime_type']['label']
        crime_description = crimes['case']['description']
        crime_nearby = crimes['nearby']

        for crime_code in crime_description:
            if '50' in crime_nearby:
                for poi_key in crime_nearby['50']:
                    if str(poi_key) not in poi_data:
                        continue
                    poi_name = str(poi_data[str(poi_key)]['sid'])
                    triple_list.append([poi_name,
                                        f'has_in_50',
                                        str(crime_code)])
                    node_set.add(poi_name)

                    triple_list.append([str(crime_key),
                                        f'has_poi_in_50',
                                        poi_name])
            if '100' in crime_nearby:
                for poi_key in crime_nearby['100']:
                    if str(poi_key) not in poi_data:
                        continue
                    poi_name = str(poi_data[str(poi_key)]['sid'])
                    triple_list.append([poi_name,
                                        f'has_in_100',
                                        str(crime_code)])
                    node_set.add(poi_name)

                    triple_list.append([str(crime_key),
                                        f'has_poi_in_100',
                                        poi_name])
            node_set.add(str(crime_code))

    triple_list = {(head, relation, tail) for head, relation, tail in triple_list}

    return list(triple_list), node_set


# One week in seconds: 604800

# min: 1262340000
# max: 1665427800

time_to_graph = {}

for idx, value in tqdm(enumerate(range(1262340000, 1665427800, 604800))):
    start = value
    end = value+604800
    district = 'Downtown'
    crime_types = ['burglary']
    poi_types = ['safety', 'government', 'company', 'restaurant', 'education']
    triples, nodes = get_snapshot(start, end, district, crime_types, poi_types)

    time_to_graph[idx] = [triples, nodes]



import pickle

with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(time_to_graph, f)

with open('saved_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

nodes_set = set()
print(len(loaded_dict))

tkg_train = []
tkg_valid = []
tkg_test = []

head_tail_str_to_int = {}
relation_str_to_int = {}

for time, (triples, nodes) in loaded_dict.items():
    for triple in triples:
        if triple[0] not in head_tail_str_to_int:
            head_tail_str_to_int[triple[0]] = len(head_tail_str_to_int)
        head = head_tail_str_to_int[triple[0]]

        if triple[2] not in head_tail_str_to_int:
            head_tail_str_to_int[triple[2]] = len(head_tail_str_to_int)
        tail = head_tail_str_to_int[triple[2]]

        if triple[1] not in relation_str_to_int:
            relation_str_to_int[triple[1]] = len(relation_str_to_int)
        relation = relation_str_to_int[triple[1]]

        if time < 532:
            tkg_train.append([head, relation, tail, time, 0])
        if time >= 532 and time < 600:
            tkg_valid.append([head, relation, tail, time, 0])
        if time > 600:
            tkg_test.append([head, relation, tail, time, 0])

    nodes_set = nodes_set.union(nodes)
    # print(f'{time}: {len(triples)}')

print(relation_str_to_int)

with open('crime_burglary.train.txt', 'w') as f:
    for line in tkg_train:
        # f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\t'+str(line[4])+'\n')
        f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

with open('crime_burglary.valid.txt', 'w') as f:
    for line in tkg_valid:
        # f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\t'+str(line[4])+'\n')
        # if str(line[1]) not in ['6', '7', '8']:
        #     continue
        f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

with open('crime_burglary.test.txt', 'w') as f:
    for line in tkg_test:
        # f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\t'+str(line[4])+'\n')
        # if str(line[1]) not in ['6', '7', '8']:
        #     continue
        f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

with open('entity2id.txt', 'w', encoding='utf-8') as f:
    for key, value in head_tail_str_to_int.items():
        f.write(str(key).replace('\n', '').replace('\t', '').replace('\r', '')+'\t'+str(value)+'\n')

with open('relation2id.txt', 'w', encoding='utf-8') as f:
    for key, value in relation_str_to_int.items():
        f.write(str(key).replace('\n', '').replace('\t', '').replace('\r', '')+'\t'+str(value)+'\n')


print(len(nodes_set))
