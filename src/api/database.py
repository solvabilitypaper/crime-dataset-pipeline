"""
    crime-dataset-pipeline

       File: database.py

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
