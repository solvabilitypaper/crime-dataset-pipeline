import logging as log
import time

from pymongo import MongoClient
from pymongo.results import BulkWriteResult, InsertManyResult, DeleteResult, UpdateResult


class DatabaseAPI:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            setattr(cls, 'instance', object.__new__(cls))
        return getattr(cls, 'instance')

    def __init__(self):
        if not hasattr(self, '_DatabaseAPI__client'):
            self.__client = MongoClient(host='localhost', port=27017)
            self.__database = self.__client['CrimeDataRepo']
            self.__district_collection = self.__database['district']
            self.__crime_collection = self.__database['crime']
            self.__street_collection = self.__database['street']
            self.__poi_collection = self.__database['poi']

    def insert(self, collection: str, entries: list) -> InsertManyResult or None:
        num_entries = len(entries)
        log.info('Write %s entries into collection "%s"', num_entries, collection)
        if num_entries == 0:
            log.info('> Skip Insert! Nothing do to.')
            return None

        start = time.time()
        collection = self.__get_collection(collection)
        result = collection.insert_many(entries)
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def select(self, collection: str, selector: dict, filter_dict: dict) -> list:
        log.info('Load collection "%s" filtered by "%s"', collection, selector)

        start = time.time()
        collection = self.__get_collection(collection)
        cursor = collection.find(selector, filter_dict)
        result = list(cursor)
        log.info('> Loaded %s documents', len(result))
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def count(self, collection: str, selector: dict):
        collection = self.__get_collection(collection)
        value = collection.count_documents(selector)

        return value

    def delete(self, collection: str, source: str) -> DeleteResult:
        log.info('Delete documents in collection "%s" filtered by "%s"', collection, source)

        start = time.time()
        collection = self.__get_collection(collection)
        result = collection.delete_many({'source': source})
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def update(self, collection: str, selector: dict, update: dict) -> UpdateResult:
        log.info('Update documents in collection "%s" filtered by "%s" with "%s"', collection, selector, update)

        start = time.time()
        collection = self.__get_collection(collection)
        result = collection.update_one(selector, update)
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def create_index(self, collection: str, field: str, unique: bool = True) -> str:
        log.info('Create a new index on field "%s" in collection "%s"', field, collection)

        start = time.time()
        collection = self.__get_collection(collection)
        result = collection.create_index(field, unique=unique)
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def bulk_write(self, collection, update_matrix: list, ordered: bool = False) -> BulkWriteResult or None:
        log.info('Performing a "Bulk Write" to collection "%s" (%s orders)', collection, len(update_matrix))

        if len(update_matrix) == 0:
            log.info('> There is nothing to do')
            return None

        start = time.time()
        collection = self.__get_collection(collection)
        result = collection.bulk_write(update_matrix, ordered=ordered)
        log.info('> Completed (Duration: %.2f seconds)', (time.time() - start))

        return result

    def drop(self) -> None:
        self.__client.drop_database('CrimeDataRepo')

    def __get_collection(self, collection):
        match collection:
            case 'district':
                collection = self.__district_collection
            case 'crime':
                collection = self.__crime_collection
            case 'street':
                collection = self.__street_collection
            case 'poi':
                collection = self.__poi_collection
            case _:
                raise KeyError

        return collection
