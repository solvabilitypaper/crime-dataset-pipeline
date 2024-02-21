import logging as log
import time

from schema import Schema, Or

from api.datasets.dataset import Dataset
from api.datasets.op_dataset import OPDataset
from api.database import DatabaseAPI


class POIData:
    def __init__(self, dataset_name: str = 'overpass'):
        self.__dataset_name = dataset_name
        self.__schema = Schema({'sid': int,
                                'source': str.lower,
                                'category': str.lower,
                                'location': {
                                    'latitude': float,
                                    'longitude': float,
                                    'street': Or(str.lower, None),
                                    'house_number': Or(str.lower, None),
                                    'city': Or(str.lower, None),
                                    'postcode': Or(int, None),
                                    'state': Or(str.lower, None),
                                    'country': Or(str.lower, None),
                                },
                                'properties': {
                                    'name': list,
                                    'phone': list,
                                    'operator': list,
                                    'description': list,
                                    'opening_hours': list,
                                    'provides': list,
                                    'link': list}
                                })

    def load_dataset(self, streets: list) -> Dataset:
        log.info('Load POI dataset: %s', self.__dataset_name)
        start = time.time()

        match self.__dataset_name:
            case 'overpass':
                dataset = OPDataset(streets)
            case _:
                raise InvalidDatasetException
        dataset.load()
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return dataset

    def apply_schema(self, dataset: Dataset) -> dict:
        log.info('Apply POI schema to %s dataset', dataset.identifier)
        start = time.time()

        if dataset.is_empty():
            raise DatasetNotLoadedException

        match dataset.identifier:
            case 'overpass':
                data = dataset.to_schema(self.__schema)
            case _:
                raise InvalidDatasetException
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return data

    def load_from_database(self) -> list:
        db_api = DatabaseAPI()
        data = db_api.select('poi', {'source': self.__dataset_name}, {})

        return data


class InvalidDatasetException(Exception):
    pass


class DatasetNotLoadedException(Exception):
    pass
