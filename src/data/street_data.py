"""*
 *     Crime Dataset Pipeline
 *
 *        File: street_data.py
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
