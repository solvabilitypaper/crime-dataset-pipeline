"""
    crime-dataset-pipeline

       File: stages.py

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

import logging as log
import time
import os
import sys

from pymongo import UpdateOne
from tqdm import tqdm

from api.database import DatabaseAPI
from api.datasets.dataset import Dataset
from constants import BAR_FORMAT, NEARBY_RADII
from data.crime_data import CrimeData
from data.district_data import DistrictData
from data.poi_data import POIData
from data.street_data import StreetData
from utils.utils import contains, handle_exception, get_distances


def init_logging():
    timestamp = int(time.time())
    log.basicConfig(filename=f'crimekg_{timestamp}.log',
                    format='%(asctime)s %(message)s (%(levelname)s)',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='w',
                    level=log.INFO)
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)
    sys.excepthook = handle_exception
    log.info('Build crime database from scratch')


def init_database() -> DatabaseAPI:
    log.info('Prepare MongoDB database')
    mdb = DatabaseAPI()
    log.info('> Drop existing database')
    mdb.drop()

    return mdb


def load_districts(mdb: DatabaseAPI):
    district_obj = DistrictData('geo_los_angeles')
    district_dataset = district_obj.load_dataset()
    district_data_list = district_obj.apply_schema(district_dataset)
    mdb.insert('district', district_data_list)
    mdb.create_index('district', 'sid', unique=True)
    mdb.create_index('district', 'properties.name', unique=True)


def load_crimes(mdb: DatabaseAPI) -> list:
    crime_obj = CrimeData('los_angeles')
    crime_dataset = crime_obj.load_dataset()
    crime_data_list = crime_obj.apply_schema(crime_dataset)
    mdb.insert('crime', crime_data_list)
    mdb.create_index('crime', 'sid', unique=True)

    return crime_data_list


def load_streets(mdb: DatabaseAPI, crime_data_list: list) -> Dataset:
    coordinates = {}
    for entry in crime_data_list:
        key = (entry['location']['latitude'], entry['location']['longitude'])
        if key not in coordinates:
            coordinates[key] = []
        coordinates[key].append(entry['sid'])

    street_obj = StreetData('open_street_map')
    street_dataset = street_obj.load_dataset(coordinates)
    street_data_dict = street_obj.apply_schema(street_dataset)
    mdb.insert('street', list(street_data_dict.values()))
    mdb.create_index('street', 'sid', unique=True)

    return street_dataset


def load_crime_database(mdb: DatabaseAPI) -> dict:
    crimes = mdb.select('crime', {'source': 'los_angeles'}, {})
    return {entry['sid']: entry for entry in crimes}


def load_street_database(mdb: DatabaseAPI) -> dict:
    streets = mdb.select('street', {'source': 'open_street_map'}, {})
    return {entry['sid']: entry for entry in streets}


def link_crime_street(mdb: DatabaseAPI,
                      crimes: dict,
                      streets: dict,
                      street_dataset: Dataset):
    log.info('Build link structure between crime and street documents')
    start = time.time()
    street_crime = {}
    crime_updates = []

    for crime_id, street_ids in tqdm(street_dataset.data['mapping'].items(), bar_format=BAR_FORMAT, desc='Crime'):
        tmp_crime_object_ids = []
        for street_id in street_ids:
            tmp_crime_object_ids.append(streets[street_id]['_id'])

            if street_id not in street_crime:
                street_crime[street_id] = []
            street_crime[street_id].append(crimes[crime_id]['_id'])

        crime_filter = {'sid': int(crime_id)}
        crime_update = {'$set': {'streets': tmp_crime_object_ids}}
        crime_updates.append(UpdateOne(crime_filter, crime_update))

    street_updates = []
    for street_id, crime_ids in tqdm(street_crime.items(), bar_format=BAR_FORMAT, desc='Street'):
        street_updates.append(UpdateOne({'sid': int(street_id)},
                                        {'$set': {'crimes': crime_ids}}))
    log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

    # Write document links to database and clean up
    mdb.bulk_write('crime', crime_updates)
    mdb.bulk_write('street', street_updates)
    crime_updates.clear()
    street_updates.clear()


def search_pois(mdb: DatabaseAPI, streets: dict) -> Dataset:
    poi_obj = POIData('overpass')
    poi_dataset = poi_obj.load_dataset(list(streets.values()))
    poi_data_dict = poi_obj.apply_schema(poi_dataset)
    mdb.insert('poi', list(poi_data_dict.values()))
    mdb.create_index('poi', 'sid', unique=True)

    return poi_dataset


def load_poi_database(mdb: DatabaseAPI):
    pois = mdb.select('poi', {'source': 'overpass'}, {})
    return {entry['sid']: entry for entry in pois}


def link_street_poi(mdb: DatabaseAPI, pois: dict, poi_dataset: Dataset):
    log.info('Build link structure between street and POI documents')
    start = time.time()
    street_updates = []

    for street_id, radii in tqdm(poi_dataset.data['mapping'].items(), bar_format=BAR_FORMAT, desc='Street'):
        street_update = {'$set': {'nearby': {}}}
        for radius, pids in radii.items():
            poi_doc_ids = [pois[pid]['_id'] for pid in pids]
            street_update['$set']['nearby'][str(radius)] = poi_doc_ids

        street_filter = {'sid': int(street_id)}
        street_updates.append(UpdateOne(street_filter, street_update))
    log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

    # Write document links to database and clean up
    mdb.bulk_write('street', street_updates)
    street_updates.clear()


def load_district_database(mdb: DatabaseAPI) -> dict:
    districts = mdb.select('district', {'source': 'geo_los_angeles'}, {})
    return {entry['sid']: entry for entry in districts}


def link_district_all(mdb: DatabaseAPI,
                      districts: dict,
                      crimes: dict,
                      streets: dict,
                      pois: dict):
    log.info('Build link structure between district and street/POI documents')
    start = time.time()
    updates = {'district': [], 'crime': [], 'poi': [], 'street': []}
    documents_dict = {'crime': crimes,
                      'poi': pois,
                      'street': streets}
    docs_with_neighborhood = set()

    for _, district in tqdm(districts.items(), bar_format=BAR_FORMAT, desc='District'):
        district_update = {'$set': {'contains': []}}
        geo_json_district = district['geometry']

        for key, documents in documents_dict.items():
            for _, document in documents.items():
                is_part_of = contains(document['location']['latitude'],
                                      document['location']['longitude'],
                                      geo_json_district)

                if not is_part_of:
                    continue

                district_update['$set']['contains'].append(document['_id'])
                doc_filter = {'sid': int(document['sid'])}
                doc_update = {'$set': {'location.neighbourhood': district['_id']}}
                updates[key].append(UpdateOne(doc_filter, doc_update))
                docs_with_neighborhood.add(str(document['_id']))

        district_filter = {'sid': district['sid']}
        updates['district'].append(UpdateOne(district_filter, district_update))

    # add empty neighborhood field to document without neighborhood
    for key, documents in documents_dict.items():
        for _, document in documents.items():
            if str(document['_id']) in docs_with_neighborhood:
                continue

            doc_filter = {'sid': int(document['sid'])}
            doc_update = {'$set': {'location.neighbourhood': None}}
            updates[key].append(UpdateOne(doc_filter, doc_update))
    log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

    # Write document links to the database and clean up
    for key, update in updates.items():
        mdb.bulk_write(key, update)
    updates.clear()
    docs_with_neighborhood.clear()
    documents_dict.clear()


def link_crime_poi(mdb: DatabaseAPI, crimes: dict, pois: dict):
    log.info('Compute distance between crime and POI locations')

    num_workers = int(os.cpu_count() / 2)
    start = time.time()
    coordinate_batches = []
    coordinates = []

    # memory: approx 20G for 10000000 pairs
    batch_size = 2500000  # batch size, empirical value

    num_pairs = len(crimes) * len(pois)
    log.info('> Number of coordination pairs to process: %s', num_pairs)

    for crime_key, crime in tqdm(crimes.items()):
        crime_point = (crime['location']['latitude'],
                       crime['location']['longitude'])

        for poi_key, poi in pois.items():
            poi_point = (poi['location']['latitude'],
                         poi['location']['longitude'])

            coordinates.append((crime_key, poi_key, crime_point, poi_point))

            if len(coordinates) == batch_size:
                coordinate_batches.append(coordinates)
                coordinates = []

            if len(coordinate_batches) == num_workers:
                distances = get_distances(coordinate_batches, num_workers)
                _link_crime_poi_store(mdb, pois, distances)
                coordinate_batches.clear()  # release memory
                distances.clear()  # release memory

    coordinate_batches.append(coordinates)
    distances = get_distances(coordinate_batches, num_workers)
    _link_crime_poi_store(mdb, pois, distances)
    coordinate_batches.clear()  # release memory
    distances.clear()  # release memory
    coordinates.clear()  # release memory
    log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))


def _link_crime_poi_store(mdb: DatabaseAPI, pois: dict, distances: list):
    log.info('> Build link structure between crime and poi documents')

    # group distances result by crime id
    nearby = {}
    for crime_key, poi_key, distance in distances:
        if crime_key not in nearby:
            nearby[crime_key] = {}

        for radius in NEARBY_RADII:
            if distance > radius:
                continue
            radius = str(radius)
            if radius not in nearby[crime_key]:
                nearby[crime_key][radius] = []
            nearby[crime_key][radius].append(pois[poi_key]['_id'])
            break

    # build the query to update the database (crime-poi-link)
    crime_updates = []
    for crime_key, values in nearby.items():
        crime_filter = {'sid': int(crime_key)}
        crime_update = {'$set': {'nearby': values}}
        crime_updates.append(UpdateOne(crime_filter, crime_update))

    # Write document links to database and clean up
    mdb.bulk_write('crime', crime_updates)
    nearby.clear()  # release memory
    crime_updates.clear()  # release memory
