import logging as log
import time

from schema import Schema, Or, Use

from api.datasets.dataset import Dataset
from api.datasets.osm_dataset import OSMDataset


class StreetData:
    def __init__(self, dataset_name: str = 'open_street_map'):
        self.__dataset_name = dataset_name
        self.__schema = Schema({'sid': int,
                                'source': str.lower,
                                'label': Or(Use(str.lower), None),
                                'category': list,
                                'location': {
                                    'road': str.lower,
                                    'city': str.lower,
                                    'county': str.lower,
                                    'state': str.lower,
                                    'country': str.lower,
                                    'country_code': str.lower,
                                    'postcode': int,
                                    'latitude': float,
                                    'longitude': float,
                                    'house_number': Or(Use(str.lower), None),
                                    'suburb': Or(Use(str.lower), None),
                                    'neighbourhood': Or(Use(str.lower), None),
                                    'above_sea_level': Or(Use(float), None),
                                    'num_lanes': Or(Use(int), None),
                                    'quarter': Or(Use(str.lower), None),
                                    'amenity': Or(Use(str.lower), None),
                                    'shop': Or(Use(str.lower), None),
                                    'town': Or(Use(str.lower), None),
                                    'residential': Or(Use(str.lower), None),
                                    'village': Or(Use(str.lower), None),
                                    'industrial': Or(Use(str.lower), None)},
                                'links': list,
                                'graph': Or(object, None)})

    def load_dataset(self, coordinates: dict) -> Dataset:
        log.info('Load street dataset: %s', self.__dataset_name)
        log.info('> Number of coordinates to process: %s', len(coordinates))
        start = time.time()

        match self.__dataset_name:
            case 'open_street_map':
                dataset = OSMDataset(coordinates)
            case _:
                raise InvalidDatasetException
        dataset.load()
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return dataset

    def apply_schema(self, dataset: Dataset) -> (list or dict):
        log.info('Apply street schema to %s dataset', dataset.identifier)
        start = time.time()

        if dataset.is_empty():
            raise DatasetNotLoadedException

        match dataset.identifier:
            case 'open_street_map':
                data = dataset.to_schema(self.__schema)
            case _:
                raise InvalidDatasetException
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return data


class InvalidDatasetException(Exception):
    pass


class DatasetNotLoadedException(Exception):
    pass
