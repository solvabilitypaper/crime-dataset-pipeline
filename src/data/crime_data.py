import logging as log
import time

from schema import Schema, Or, Use, And

from api.datasets.dataset import Dataset
from api.datasets.la_dataset import LADataset
from api.database import DatabaseAPI


class CrimeData:
    def __init__(self, dataset_name: str = 'los_angeles'):
        self.__dataset_name = dataset_name
        schema_dict = {'sid': int,
                       'source': str,
                       'timestamp': {'reported': float,
                                     'occurred': float},
                       'location': {'area_id': int,
                                    'area_label': str.lower,
                                    'street': str.lower,
                                    'cross_street': Or(Use(str.lower), None),
                                    'latitude': float,
                                    'longitude': float,
                                    'place_id': Or(Use(int), None),
                                    'place_label': Or(Use(str.lower), None)},
                       'victim': {'age': Or(And(Use(int),
                                                lambda n: 4 <= n <= 110),
                                            None),
                                  'gender': Or(Use(str.lower), None),
                                  'descent': Or(Use(str.lower), None)},
                       'weapon': {'id': Or(Use(int), None),
                                  'label': Or(Use(str.lower), None)},
                       'crime_type': {'id': int,
                                      'sub_ids': list,
                                      'label': str.lower},
                       'case': {'id': Or(Use(str.lower), None),
                                'label': Or(Use(str.lower), None),
                                'description': dict}}
        self.__schema = Schema(schema_dict)

    def load_dataset(self) -> Dataset:
        log.info('Load crime dataset: %s', self.__dataset_name)
        start = time.time()

        match self.__dataset_name:
            case 'los_angeles':
                dataset = LADataset()
            case _:
                raise InvalidDatasetException
        dataset.load()
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return dataset

    def apply_schema(self, dataset: Dataset) -> list:
        log.info('Apply crime schema to %s dataset', dataset.identifier)
        start = time.time()

        if dataset.is_empty():
            raise DatasetNotLoadedException

        match dataset.identifier:
            case 'los_angeles':
                data = dataset.to_schema(self.__schema)
            case _:
                raise InvalidDatasetException
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return data

    def load_from_database(self) -> list:
        db_api = DatabaseAPI()
        data = db_api.select('crime', {'source': self.__dataset_name}, {})

        return data


class InvalidDatasetException(Exception):
    pass


class DatasetNotLoadedException(Exception):
    pass
