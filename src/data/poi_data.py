"""*
 *     Crime Dataset Pipeline
 *
 *        File: poi_data.py
 *
 *     Authors: Deleted for purposes of anonymity
 *
 *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
 *
 * The software and its source code contain valuable trade secrets and shall be maintained in
 * confidence and treated as confidential information. The software may only be used for
 * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
 * license agreement or nondisclosure agreement with the proprietor of the software.
 * Any unauthorized publication, transfer to third parties, or duplication of the object or
 * source code---either totally or in part---is strictly prohibited.
 *
 *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
 *     All Rights Reserved.
 *
 * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
 * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
 *
 * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
 * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
 * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
 * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
 * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
 * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
 * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGES.
 *
 * For purposes of anonymity, the identity of the proprietor is not given herewith.
 * The identity of the proprietor will be given once the review of the
 * conference submission is completed.
 *
 * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 *"""

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
