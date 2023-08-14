"""
    crime-dataset-pipeline

       File: geo_la_dataset.py

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
