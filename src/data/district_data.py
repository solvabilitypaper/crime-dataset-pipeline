import logging as log
import time

from schema import Schema

from api.datasets.dataset import Dataset
from api.datasets.geo_la_dataset import GeoLADataset
from api.database import DatabaseAPI


class DistrictData:
    def __init__(self, dataset_name: str = 'geo_los_angeles'):
        self.__dataset_name = dataset_name
        self.__schema = Schema({'sid': str,
                                'source': str,
                                'properties': {
                                    'name': str.lower},
                                'geometry': {
                                    'type': str.lower,
                                    'coordinates': list}})

    def load_dataset(self) -> Dataset:
        log.info('Load district dataset: %s', self.__dataset_name)
        start = time.time()

        match self.__dataset_name:
            case 'geo_los_angeles':
                dataset = GeoLADataset()
            case _:
                raise InvalidDatasetException
        dataset.load()
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return dataset

    def apply_schema(self, dataset: Dataset) -> list:
        log.info('Apply district schema to %s dataset', dataset.identifier)
        start = time.time()

        if dataset.is_empty():
            raise DatasetNotLoadedException

        match dataset.identifier:
            case 'geo_los_angeles':
                data = dataset.to_schema(self.__schema)
            case _:
                raise InvalidDatasetException
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return data

    def load_from_database(self) -> list:
        db_api = DatabaseAPI()
        data = db_api.select('district', {'source': self.__dataset_name}, {})

        return data


class InvalidDatasetException(Exception):
    pass


class DatasetNotLoadedException(Exception):
    pass
