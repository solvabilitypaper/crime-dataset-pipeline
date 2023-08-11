import logging as log
import os
from pathlib import Path

import geojson
from schema import Schema, SchemaError
from tqdm import tqdm

from api.datasets.dataset import Dataset
from constants import BAR_FORMAT


class GeoLADataset(Dataset):
    def __init__(self):
        self.identifier = 'geo_los_angeles'
        self.__dataset_file = 'LA_Neighborhood_Boundaries.geojson'

    def load(self):
        self.__read_files()

    def to_schema(self, schema: Schema) -> list:
        data = []
        features = self.data['features']
        num_features = len(features)

        for row in tqdm(features, total=num_features, bar_format=BAR_FORMAT):
            entry = {'sid': row['properties']['external_i'],
                     'source': self.identifier,
                     'properties': {
                         'name': row['properties']['name']},
                     'geometry': {
                         'type': row['geometry']['type'],
                         'coordinates': row['geometry']['coordinates']}}

            try:
                validated = schema.validate(entry)
            except SchemaError:
                log.error('> District schema not applicable: %s',
                          row['properties']['external_i'])
                continue

            data.append(validated)

        return data

    def __read_files(self):
        filename = self.__dataset_file
        root = Path(__file__).parent.parent.parent.parent.resolve()
        file_path = os.path.join(root, 'data', self.identifier, filename)
        log.info('> Loading: %s', file_path)

        with open(file_path, 'r', encoding='utf-8') as file_obj:
            self.data = geojson.load(file_obj)
