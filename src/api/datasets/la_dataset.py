"""
    crime-dataset-pipeline

       File: la_dataset.py

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

import numpy as np
import pandas as pd
from schema import Schema, SchemaError
from tqdm import tqdm

from utils import utils
from api.datasets.dataset import Dataset
from constants import BAR_FORMAT


class LADataset(Dataset):
    def __init__(self):
        self.identifier = 'los_angeles'
        self.__dataset_files = ['Crime_Data_from_2010_to_2019.csv',
                                'Crime_Data_from_2020_to_Present.csv']
        self.__crime_codes_file = 'MO_CODES_Numerical_20191119.txt'
        self.data = pd.DataFrame({})

    def load(self):
        self.__read_files()
        self.__parse_area_column()
        self.__parse_age_column()
        self.data = self.data.replace(np.nan, None)  # replace nan with None
        self.__normalize()
        self.__parse_date_columns()
        self.__parse_victsex_column()
        self.__parse_victdescent_column()
        self.__read_crime_codes()

    def to_schema(self, schema: Schema) -> list:
        data = []
        entries = self.data.iterrows()
        num_entries = len(self.data.index)

        for _, row in tqdm(entries, total=num_entries, bar_format=BAR_FORMAT):
            entry = {'sid': row['DR_NO'],
                     'source': self.identifier,
                     'timestamp': {'reported': row['Date Rptd'],
                                   'occurred': row['DATE OCC']},
                     'location': {'area_id': row['AREA'],
                                  'area_label': row['AREA NAME'],
                                  'street': row['LOCATION'],
                                  'cross_street': row['Cross Street'],
                                  'latitude': row['LAT'],
                                  'longitude': row['LON'],
                                  'place_id': row['Premis Cd'],
                                  'place_label': row['Premis Desc']},
                     'victim': {'age': row['Vict Age'],
                                'gender': row['Vict Sex'],
                                'descent': row['Vict Descent']},
                     'weapon': {'id': row['Weapon Used Cd'],
                                'label': row['Weapon Desc']},
                     'crime_type': {'id': row['Crm Cd'],
                                    'sub_ids': [row['Crm Cd 1'],
                                                row['Crm Cd 2'],
                                                row['Crm Cd 3'],
                                                row['Crm Cd 4']],
                                    'label': row['Crm Cd Desc']},
                     'case': {'id': row['Status'],
                              'label': row['Status Desc'],
                              'description': self.__schema_case_desc(row)}}

            # TODO: Add the ones below
            # row['Rpt Dist No']
            # row['Part 1-2']

            try:
                validated = schema.validate(entry)
            except SchemaError:
                log.error('> Crime schema not applicable: %s', row['DR_NO'])
                continue

            data.append(validated)

        return data

    def __read_files(self):
        data_list = []

        for filename in self.__dataset_files:
            root = Path(__file__).parent.parent.parent.parent.resolve()
            file_path = os.path.join(root, 'data', self.identifier, filename)
            log.info('> Loading: %s', file_path)
            data_list.append(pd.read_csv(file_path))

        self.data = pd.concat(data_list)

    def __parse_area_column(self):
        log.info('> Parse area columns')
        self.data['AREA'] = self.data['AREA'].fillna(self.data['AREA '])
        self.data = self.data.drop(['AREA '], axis=1)

    def __parse_age_column(self):
        log.info('> Parse age columns')
        self.data.loc[self.data['Vict Age'] > 110, 'Vict Age'] = np.nan
        self.data.loc[self.data['Vict Age'] < 4, 'Vict Age'] = np.nan

    def __normalize(self):
        log.info('> Normalize columns')
        # convert float to int
        columns_int = ['AREA', 'Premis Cd', 'Weapon Used Cd', 'Crm Cd',
                       'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']
        for key in columns_int:
            self.data[key] = utils.normalize_column_int(self.data[key])

        # normalize string values
        columns_str = ['AREA NAME', 'LOCATION', 'Cross Street', 'Premis Desc',
                       'Vict Sex', 'Vict Descent', 'Weapon Desc', 'Status',
                       'Crm Cd Desc', 'Status Desc', 'TIME OCC']
        for key in columns_str:
            self.data[key] = utils.normalize_column_str(self.data[key])

    def __parse_date_columns(self):
        log.info('> Parse date columns')
        # replace broken value by 12:00:00. There is no reliable way to fix it.
        self.data['TIME OCC'] = np.where(self.data['TIME OCC'].str.len() != 4,
                                         '12:00:00',
                                         self.data['TIME OCC'].str[0:2] + ':' + self.data['TIME OCC'].str[2:4] + ':00')

        # merge columns: DATE OCC, TIME OCC
        str_pos = self.data['DATE OCC'].str.find(' ') + 1
        date_pos = zip(self.data['DATE OCC'], str_pos)
        self.data['DATE OCC'] = [value[:pos] for value, pos in date_pos]
        self.data['DATE OCC'] = self.data['DATE OCC'] + self.data['TIME OCC']

        # convert date strings to timestamp object
        columns_date = [('DATE OCC', '%m/%d/%Y %H:%M:%S'),
                        ('Date Rptd', '%m/%d/%Y %H:%M:%S %p')]
        for column_date, regex_date in columns_date:
            self.data[column_date] = utils.normalize_column_date(self.data[column_date], regex_date)

    def __parse_victsex_column(self):
        log.info('> Parse gender columns')
        # According to the documentation, there are only three valid values:
        # male, female, and unknown
        gender_dict = {'f': 'female',
                       'm': 'male',
                       r'[^fm]': None}
        self.data['Vict Sex'] = self.data['Vict Sex'].astype(str).str.lower().replace(regex=gender_dict)

    def __parse_victdescent_column(self):
        log.info('> Parse descent columns')
        descent_dict = {'a': 'other asian',
                        'b': 'black',
                        'c': 'chinese',
                        'd': 'cambodian',
                        'f': 'filipino',
                        'g': 'guamanian',
                        'h': 'hispanic-latin-mexican',
                        'i': 'american_indian-alaskan_native',
                        'j': 'japanese',
                        'k': 'korean',
                        'l': 'laotian',
                        'o': 'other',
                        'p': 'pacific islander',
                        's': 'samoan',
                        'u': 'hawaiian',
                        'v': 'vietnamese',
                        'w': 'white',
                        'x': None,
                        'z': 'asian indian',
                        '-': None,
                        r'[^abcdfghijklopsuvwxz-]': None}
        self.data['Vict Descent'] = self.data['Vict Descent'].astype(str).str.lower().replace(regex=descent_dict)

    def __read_crime_codes(self):
        root = Path(__file__).parent.parent.parent.parent.resolve()
        file_path = os.path.join(root,
                                 'data',
                                 self.identifier,
                                 self.__crime_codes_file)
        log.info('> Parse: %s', file_path)

        with open(file_path, encoding='utf-8') as file:
            lines = [line.rstrip() for line in file]
        code_to_text = {str(lines[idx]): lines[idx + 1]
                        for idx in range(0, len(lines) - 1, 2)}

        mocodes_text = []
        for row in self.data['Mocodes'].tolist():
            if row is None:
                mocodes_text.append(None)
                continue

            code_desc = [code_to_text[code] if code in code_to_text else ''
                         for code in row.split()]
            mocodes_text.append(';'.join(code_desc))

        self.data['Mocodes_Text'] = mocodes_text

    @staticmethod
    def __schema_case_desc(row: pd.Series) -> dict:
        code_to_desc = {}

        if row['Mocodes'] is not None:
            code_desc = row['Mocodes_Text'].split(';')
            for idx, code in enumerate(row['Mocodes'].split()):
                code_to_desc[code] = code_desc[idx]

        return code_to_desc
