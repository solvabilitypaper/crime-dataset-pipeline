import logging as log
import time

from schema import Schema, SchemaError
from tqdm import tqdm

from api.datasets.dataset import Dataset
from utils import utils
from constants import BAR_FORMAT


class OSMDataset(Dataset):

    def __init__(self, coordinates: dict):
        self.identifier = 'open_street_map'
        self.__coordinates = coordinates
        self.__reverse_url = 'https://nominatim.openstreetmap.org/reverse?'

    def load(self):
        source_to_street, street_dict = self.__reverse_geocoding()
        self.data = {'mapping': source_to_street, 'street': street_dict}

        self.__remove_broken_entries()
        self.__add_placeholders()
        self.__normalize()

    def to_schema(self, schema: Schema) -> dict:
        data = {}
        entries = self.data['street'].items()

        for key, value in tqdm(entries, bar_format=BAR_FORMAT):
            entry = {'sid': value['osm_id'],
                     'source': self.identifier,
                     'label': value['name'],
                     'category': list({value['addresstype'],
                                       value['category'],
                                       value['type']}),
                     'location': {
                         'road': value['address']['road'],
                         'city': value['address']['city'],
                         'county': value['address']['county'],
                         'state': value['address']['state'],
                         'country': value['address']['country'],
                         'country_code': value['address']['country_code'],
                         'postcode': value['address']['postcode'],
                         'latitude': value['lat'],
                         'longitude': value['lon'],
                         'house_number': value['address']['house_number'],
                         'suburb': value['address']['suburb'],
                         'neighbourhood': value['address']['neighbourhood'],
                         'above_sea_level': value['extratags']['ele'],
                         'num_lanes': value['extratags']['lanes'],
                         'quarter': value['address']['quarter'],
                         'amenity': value['address']['amenity'],
                         'shop': value['address']['shop'],
                         'town': value['address']['town'],
                         'residential': value['address']['residential'],
                         'village': value['address']['village'],
                         'industrial': value['address']['industrial']},
                     'links': list({value['extratags']['website'],
                                    value['extratags']['contact:website']}),
                     'graph': None}

            entry['category'] = [value for value in entry['category'] if value]
            entry['links'] = [value for value in entry['links'] if value]

            try:
                validated = schema.validate(entry)
            except SchemaError:
                log.error('> Street schema not applicable: %s', key)
                continue
            data[key] = validated

        return data

    def __reverse_geocoding(self) -> (dict, dict):
        log.info('> Start reverse geocoding')
        street_dict = {}
        source_to_street = {}
        num_skipped = 0
        max_attempts = 10

        for (latitude, longitude), sids in tqdm(self.__coordinates.items(), bar_format=BAR_FORMAT):
            params = {'lat': latitude,
                      'lon': longitude,
                      'format': 'jsonv2',
                      'extratags': 1}

            response = None
            wait_time = 5  # seconds
            for _ in range(max_attempts):
                response = utils.handle_request(self.__reverse_url, params)
                if response:
                    break

                log.warning('>> Invalid response: %s, %s', latitude, longitude)
                log.info('>> Sleep for %s seconds then retry',  wait_time)
                time.sleep(wait_time)
                wait_time *= 2

            if not response:
                num_skipped += 1
                continue

            response = response.json()

            if response['osm_id'] not in street_dict:
                street_dict[response['osm_id']] = response
            for sid in sids:
                if sid not in source_to_street:
                    source_to_street[sid] = []
                source_to_street[sid].append(response['osm_id'])

        if num_skipped > 0:
            log.info('>> Could not reverse geocode %s entries', num_skipped)

        return source_to_street, street_dict

    def __remove_broken_entries(self):
        log.info('> Remove broken entries')

        # check for mandatory fields
        fields = ['lat', 'lon', 'address']
        fields_address = ['road', 'city', 'county', 'state',
                          'country', 'country_code', 'postcode']
        to_be_deleted = set()
        for key, value in self.data['street'].items():
            for field in fields:
                if field not in value:
                    to_be_deleted.add(key)
                    break

            if 'address' not in value:
                continue
            address_data = value['address']

            for field_address in fields_address:
                if field_address not in address_data or \
                        (field_address == 'postcode'
                         and 'postcode' in address_data
                         and not utils.is_integer(address_data['postcode'])):
                    to_be_deleted.add(key)
                    break

        # delete broken entries
        for key_value in to_be_deleted:
            del self.data['street'][key_value]

        # clean up mapping
        for key, value in self.data['mapping'].items():
            to_be_discarded = []
            for key_value in value:
                if key_value in to_be_deleted:
                    to_be_discarded.append(key_value)
            for outdated_value in to_be_discarded:
                value.remove(outdated_value)

        log.info('>> Removed %s entries', len(to_be_deleted))

    def __add_placeholders(self):
        log.info('> Add placeholders for missing fields')

        # fields that may not exist
        fields = ['name', 'addresstype', 'category', 'type']
        fields_address = ['house_number', 'suburb', 'neighbourhood', 'quarter',
                          'amenity', 'shop', 'town', 'residential', 'village',
                          'industrial']
        fields_extras = ['website', 'contact:website', 'lanes', 'ele']

        # create field with an empty value
        for _, data in self.data['street'].items():
            for key in fields:
                if key not in data or not data[key]:
                    data[key] = 'none'
            for key in fields_address:
                if key not in data['address'] or not data['address'][key]:
                    data['address'][key] = 'none'
            for key in fields_extras:
                if key not in data['extratags'] or not data['extratags'][key]:
                    data['extratags'][key] = 'none'

    def __normalize(self):
        log.info('> Normalize values')

        fields = {'int': ['osm_id'],
                  'float': ['lat', 'lon'],
                  'str': ['name', 'addresstype', 'category', 'type'],
                  'address_int': ['postcode'],
                  'address_str': ['road', 'city', 'county', 'state', 'country',
                                  'country_code', 'house_number', 'quarter',
                                  'neighbourhood', 'amenity', 'shop', 'town',
                                  'residential', 'village', 'industrial'],
                  'extra_str': ['website', 'contact:website'],
                  'extra_int': ['lanes'],
                  'extra_float': ['ele']}

        for _, value in self.data['street'].items():
            # normalize int values
            for key in fields['int']:
                value[key] = utils.normalize_int(value[key])

            # normalize float values
            for key in fields['float']:
                value[key] = utils.normalize_float(value[key])

            # normalize string values
            for key in fields['str']:
                value[key] = utils.normalize_str(value[key])

            # normalize street int values
            for key in fields['address_int']:
                tmp = utils.normalize_int(value['address'][key])
                value['address'][key] = tmp

            # normalize street string values
            for key in fields['address_str']:
                tmp = utils.normalize_str(value['address'][key])
                value['address'][key] = tmp

            # normalize extra tags string values
            for key in fields['extra_str']:
                tmp = utils.normalize_str(value['extratags'][key])
                value['extratags'][key] = tmp

            # normalize extra tags int values
            for key in fields['extra_int']:
                tmp = utils.normalize_int(value['extratags'][key])
                value['extratags'][key] = tmp

            # normalize extra tags float values
            for key in fields['extra_float']:
                tmp = utils.normalize_float(value['extratags'][key])
                value['extratags'][key] = tmp
