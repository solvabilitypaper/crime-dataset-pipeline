import logging as log
import time

import overpass
from overpass.errors import ServerLoadError, UnknownOverpassError, MultipleRequestsError
from schema import Schema, SchemaError
from tqdm import tqdm

from api.datasets.dataset import Dataset
from utils import utils
from constants import BAR_FORMAT, NEARBY_RADII


class OPDataset(Dataset):
    def __init__(self, streets: list):
        self.identifier = 'overpass'
        self.data = {'mapping': {}, 'poi': {}}

        self.__streets = streets

    def load(self):
        self.__nearby()
        self.__sort_by_category()

        self.__normalize()
        self.__repair()
        self.__cluster_properties()

    def __nearby(self):
        log.info('> Search nearby locations for street documents')
        api = overpass.API()

        for entry in tqdm(self.__streets, bar_format=BAR_FORMAT):
            latitude = entry['location']['latitude']
            longitude = entry['location']['longitude']
            wait_time = 5

            result = {}
            for radius in NEARBY_RADII:
                request = f'node(around:{radius},{latitude},{longitude})'
                while True:
                    try:
                        response = api.Get(request)
                        break
                    except (ServerLoadError,
                            UnknownOverpassError,
                            MultipleRequestsError,
                            UnboundLocalError) as error_message:
                        if wait_time > 120:
                            log.warning('>> %s', error_message)
                            log.info('>> Sleep for %s seconds', wait_time)
                        time.sleep(wait_time)
                        wait_time *= 2

                result[radius] = response

            self.__filter_and_store(entry['sid'], result)

        log.info('>> Found %s locations', len(self.data['poi']))

    def __filter_and_store(self, street_id: str, values: dict):
        # These were manually selected
        fields = ['residential', 'facebook', 'instagram', 'books', 'name',
                  'public_transport', 'emergency', 'stationery', 'beauty',
                  'category', 'vending', 'museum', 'hairdresser', 'railway',
                  'location', 'parcel_mail_in', 'drive_in', 'erotic', 'pub',
                  'contact:*', 'video', 'repair', 'office', 'social_facility',
                  'club', 'healthcare', 'wine', 'food', 'baseball', 'cuisine',
                  'music', 'product', 'opening_hours:*', 'water', 'company',
                  'cultivar', 'bbq', 'bunker_type', 'artwork_type', 'seats',
                  'brewery', 'railway:*', 'wayside_shrine', 'waste',
                  'gay', 'backrest', 'website', 'manufacturer', 'military',
                  'memorial', 'fitness_station', 'type', 'clothes', 'subway',
                  'beer', 'swimming_pool', 'studio', 'tram', 'bus', 'place',
                  'tourism', 'theatre:*', 'landmark', 'pet', 'pumping_station',
                  'substation', 'golf', 'branch', 'craft', 'statue', 'service',
                  'advertising', 'furniture', 'garden', 'live_music', 'dock',
                  'light_rail', 'takeaway', 'bicycle_rental', 'sport',
                  'enforcement', 'playground', 'education', 'station', 'train',
                  'toilets', 'military_service', 'fountain', 'maxspeed', 'atm',
                  'school', 'ice_cream', 'building', 'cemetery', 'amenity',
                  'lgbtq', 'urgent_care', 'aerialway', 'cocktails', 'religion',
                  'drive_through', 'bench', 'parking', 'microbrewery', 'lit',
                  'karaoke', 'kiosk', 'twitter', 'stars', 'tobacco', 'natural',
                  'consulate', 'wikipedia', 'nursery', 'preschool', 'horse',
                  'waterway', 'dog', 'diplomatic', 'school_type', 'phone',
                  'surveillance:*', 'coffee', 'rental', 'education_profile',
                  'shop', 'bar', 'park_ride']
        fields = list(map(str.lower, fields))

        # These were manually selected
        ignores = {('railway', 'level_crossing'), ('railway', 'switch'),
                   ('natural', 'tree'), ('railway', 'crossing'),
                   ('amenity', 'parking_entrance'), ('amenity', 'post_box'),
                   ('amenity', 'letter_box'), ('amenity', 'recycling'),
                   ('railway', 'railway_crossing'), ('disused:*', '*'),
                   ('construction:*', '*'), ('demolished:*', '*'),
                   ('old_shop', '*'), ('barrier', '*'), ('natural', 'shrub'),
                   ('railway', 'buffer_stop'), ('construction', '*'),
                   ('natural', 'peak'), ('fixme', '*'), ('amenity', 'clock'),
                   ('amenity', 'loading_dock'), ('amenity', 'waste_disposal'),
                   ('ant_sys_ref', '*'), ('railway', 'derail'),
                   ('golf', 'pin'), ('golf', 'hole'), ('was:*', '*'),
                   ('highway', 'street_lamp'), ('railway', 'owner_change'),
                   ('railway', 'tram_level_crossing'), ('natural', 'spring'),
                   ('railway', 'signal'), ('railway', 'tram_crossing'),
                   ('amenity', 'lounger'), ('amenity', 'watering_place'),
                   ('amenity', 'animal_shelter'), ('man_made', 'telescope'),
                   ('name', 'Crosswalk')}

        street_to_pid = self.data['mapping']
        pid_to_poi = self.data['poi']

        if street_id not in street_to_pid:
            street_to_pid[street_id] = {}

        for radius, points in values.items():
            if radius not in street_to_pid[street_id]:
                street_to_pid[street_id][radius] = set()

            for point in points['features']:
                if 'properties' not in point:
                    continue
                list_keys = point['properties'].keys()
                list_keys = set(map(str.lower, list_keys))

                if self.__matches(point['properties'], ignores):
                    continue

                for key in fields:
                    for value in list_keys:
                        if key == value or (key.endswith(':*') and value.startswith(key[:-2])):
                            pid_to_poi[point['id']] = point
                            street_to_pid[street_id][radius].add(point['id'])

        for _, radii in street_to_pid.items():
            sorted_radii = sorted(list(radii.keys()))
            for idx in range(1, len(sorted_radii)):
                known_pois = [radii[sorted_radii[tmp]] for tmp in range(idx)]
                known_pois = set().union(*known_pois)
                radius = sorted_radii[idx]
                radii[radius] = radii[radius] - known_pois

        self.data = {'mapping': street_to_pid, 'poi': pid_to_poi}

    def __sort_by_category(self):
        category_to_pid = {'restaurant': set(),  # restaurant / bar / club
                           'transport': set(),  # public transport
                           'health_care': set(),  # health_care
                           'education': set(),  # school
                           'tourism': set(),  # culture / tourism / recreation
                           'finance': set(),  # finance
                           'sport': set(),  # sport
                           'shopping': set(),  # shopping
                           'entertainment': set(),
                           'company': set(),  # company / services
                           'organization': set(),
                           'safety': set(),  # military
                           'area': set(),  # district, places
                           'government': set(),  # government / public service
                           'public': set(),  # parks
                           'other': set()}  # residential

        for pid, poi in self.data['poi'].items():
            properties = poi['properties']

            if self.__is_restaurant(properties):
                category_to_pid['restaurant'].add(pid)
            elif self.__is_transport(properties):
                category_to_pid['transport'].add(pid)
            elif self.__is_healthcare(properties):
                category_to_pid['health_care'].add(pid)
            elif self.__is_education(properties):
                category_to_pid['education'].add(pid)
            elif self.__is_tourism(properties):
                category_to_pid['tourism'].add(pid)
            elif self.__is_finance(properties):
                category_to_pid['finance'].add(pid)
            elif self.__is_sport(properties):
                category_to_pid['sport'].add(pid)
            elif self.__is_shopping(properties):
                category_to_pid['shopping'].add(pid)
            elif self.__is_entertainment(properties):
                category_to_pid['entertainment'].add(pid)
            elif self.__is_company(properties):
                category_to_pid['company'].add(pid)
            elif self.__is_organization(properties):
                category_to_pid['organization'].add(pid)
            elif self.__is_safety(properties):
                category_to_pid['safety'].add(pid)
            elif self.__is_area(properties):
                category_to_pid['area'].add(pid)
            elif self.__is_government(properties):
                category_to_pid['government'].add(pid)
            elif self.__is_public(properties):
                category_to_pid['public'].add(pid)
            elif self.__is_other(properties):
                category_to_pid['other'].add(pid)
            else:
                category_to_pid['other'].add(pid)

        self.data['category_to_pid'] = category_to_pid

    def __is_restaurant(self, entry: dict) -> bool:
        patterns = {('amenity', 'restaurant'), ('amenity', 'bar'),
                    ('amenity', 'cafe'), ('amenity', 'pub'),
                    ('amenity', 'fast_food'), ('amenity', 'nightclub'),
                    ('amenity', 'ice_cream'), ('amenity', 'food_court'),
                    ('takeaway', 'yes'), ('bar', '*'), ('brewery', '*'),
                    ('club', '*'), ('name', 'Seven Stars Creation'),
                    ('name', 'coffee window')}
        ignores = {('tourism', '*'), ('club', 'sport'),
                   ('leisure', 'amusement_arcade')}

        return self.__matches(entry, patterns, ignores)

    def __is_transport(self, entry: dict) -> bool:
        patterns = {('public_transport', 'platform'), ('railway', 'stop'),
                    ('railway', 'subway_entrance'), ('railway', 'station'),
                    ('public_transport', 'stop_position'), ('amenity', 'taxi'),
                    ('highway', 'bus_stop'), ('amenity', 'motorcycle_rental'),
                    ('gnis:feature_type', 'airport'), ('bicycle_parking', '*'),
                    ('amenity', 'bicycle_parking'), ('amenity', 'bus_station'),
                    ('amenity', 'charging_station'), ('name', '*parking*'),
                    ('amenity', 'bicycle_rental'), ('amenity', 'car_pooling'),
                    ('operator', 'Pacific Harbor Line'),
                    ('amenity', 'parking_space'), ('amenity', 'car_rental'),
                    ('railway', 'spur_junction'), ('amenity', 'parking'),
                    ('amenity', 'escooter_rental'), ('amenity', 'car_sharing'),
                    ('office', 'flight_services'), ('service', 'bicycle:diy'),
                    ('amenity', 'scooter_share'), ('type', 'public_transport'),
                    ('aerialway', 'station'), ('bicycle_rental', '*'),
                    ('transport', '*')}
        ignores = {('name', '*plaza*'), ('name', '*college*'),
                   ('office', 'government'), ('name', '*hotel*'),
                   ('surveillance:type', 'camera'), ('shop', '*'),
                   ('name', '*school*'), ('name', 'union & court')}

        return self.__matches(entry, patterns, ignores)

    def __is_healthcare(self, entry: dict) -> bool:
        patterns = {('amenity', 'pharmacy'), ('office', 'therapist'),
                    ('healthcare', '*'), ('amenity', 'clinic'),
                    ('name', '*health*'), ('amenity', 'veterinary'),
                    ('amenity', 'veterinary_pharmacy'), ('name', 'kushfly'),
                    ('name', '*medical staffing*'), ('office', 'healthcare'),
                    ('amenity', 'dentist'), ('amenity', 'doctors'),
                    ('name', '*dental*')}
        ignores = {('shop', '*'), ('amenity', 'restaurant'),
                   ('name', 'occidental'), ('name', '*Research Center*'),
                   ('operator', '*metropolitan transportation authority*'),
                   ('amenity', 'place_of_worship')}

        return self.__matches(entry, patterns, ignores)

    def __is_education(self, entry: dict) -> bool:
        patterns = {('animal', 'school'), ('amenity', 'animal_training'),
                    ('amenity', 'school'), ('amenity', 'research_institute'),
                    ('preschool', 'yes'), ('amenity', 'kindergarten'),
                    ('amenity', 'music_school'), ('amenity', 'driving_school'),
                    ('office', 'educational_institution'), ('education', '*'),
                    ('amenity', 'prep_school'), ('name', '*training*'),
                    ('education_profile', '*'), ('dance:teaching', '*',),
                    ('name', '*school*'), ('building', 'university'),
                    ('amenity', 'university'), ('amenity', 'training'),
                    ('amenity', 'college'), ('name', '*college*'),
                    ('amenity', 'language_school'), ('office', 'research'),
                    ('name', '*research center*'), ('amenity', 'childcare'),
                    ('name', '*children\'s center*'), ('school_type', '*'),
                    ('name', '*dimas-alang*'), ('school', '*')}
        ignores = {('leisure', 'fitness_centre'), ('amenity', 'bar'),
                   ('amenity', 'bicycle_rental'), ('amenity', 'restaurant'),
                   ('leisure', 'sports_centre'), ('office', 'company')}

        return self.__matches(entry, patterns, ignores)

    def __is_tourism(self, entry: dict) -> bool:
        patterns = {('artwork_type', '*'), ('tourism', '*'), ('museum', '*'),
                    ('cultivar', '*'), ('dock', '*'),
                    ('name', '*hotel*'), ('name', 'Wilshire Grand Center'),
                    ('office', 'travel_agent'), ('amenity', 'arts_centre'),
                    ('name', 'Halliburton Building'), ('name', 'sika'),
                    ('name', 'Hite Building'), ('name', 'gallery plus'),
                    ('name', 'World Stage Performance Gallery'),
                    ('office', 'tourist_bus'), ('historic', '*')}
        ignores = {('amenity', 'restaurant'), ('amenity', 'bicycle_rental'),
                   ('historic', 'district'), ('shop', '*'),
                   ('name', '*plaza*'), ('amenity', 'bar'),
                   ('office', 'company'), ('amenity', 'animal_boarding'),
                   ('name', '*fire station*'), ('amenity', 'place_of_worship'),
                   ('gnis:feature_type', 'airport'), ('amenity', 'shelter')}

        return self.__matches(entry, patterns, ignores)

    def __is_finance(self, entry: dict) -> bool:
        patterns = {('amenity', 'bank'), ('office', 'financial_advisor'),
                    ('amenity', 'atm'), ('name', '*stock exchange*'),
                    ('name', '*gold buyer*'), ('atm', '*'),
                    ('office', 'tax_advisor'), ('office', 'bail_bond_agent'),
                    ('amenity', 'bureau_de_change'), ('office', 'financial'),
                    ('amenity', 'money_transfer'), ('office', 'accountant'),
                    ('name', '*bank building*'), ('office', 'finance')}
        ignores = {('atm', 'yes')}

        return self.__matches(entry, patterns, ignores)

    def __is_sport(self, entry: dict) -> bool:
        patterns = {('leisure', 'fitness_centre'), ('leisure', 'sports_centre'),
                    ('amenity', 'dojo'), ('baseball', '*'), ('sport', '*'),
                    ('swimming_pool', '*'), ('leisure', 'sports_hall')}
        ignores = {('amenity', 'pub'), ('artwork_type', 'statue'),
                   ('leisure', 'park')}

        return self.__matches(entry, patterns, ignores)

    def __is_shopping(self, entry: dict) -> bool:
        patterns = {('amenity', 'fuel'), ('amenity', 'bicycle_repair_station'),
                    ('amenity', 'car_wash'), ('beauty', '*'), ('vending', '*'),
                    ('advertising', '*'), ('amenity', 'post_office'),
                    ('amenity', 'vending_machine'), ('landuse', 'retail'),
                    ('name', '*africa by the yard*'), ('craft', '*'),
                    ('name', '*african urban wear*'), ('shop', '*')}
        ignores = {('amenity', 'cafe'),
                   ('sport', '*'), ('amenity', 'bench'), ('amenity', 'bar'),
                   ('amenity', 'restaurant'), ('leisure', 'dance'),
                   ('amenity', 'training'), ('amenity', 'events_venue'),
                   ('amenity', 'fast_food'), ('craft', 'signmaker'),
                   ('craft', 'plumber'), ('amenity', 'bank')}

        return self.__matches(entry, patterns, ignores)

    def __is_entertainment(self, entry: dict) -> bool:
        patterns = {('amenity', 'cinema'), ('description', 'concert hall'),
                    ('leisure', 'bowling_alley'), ('amenity', 'photo_booth'),
                    ('theatre:type', '*'), ('amenity', 'music_venue'),
                    ('amenity', 'events_venue'), ('leisure', 'sauna'),
                    ('amenity', 'theatre'), ('name', '*imax*'),
                    ('leisure', 'amusement_arcade'), ('amenity', 'spa'),
                    ('amenity', 'planetarium'), ('amenity', 'karaoke_box'),
                    ('leisure', 'escape_game'), ('leisure', 'indoor_play'),
                    ('leisure', 'adult_gaming_centre'), ('playground', '*'),
                    ('leisure', 'playground'), ('amenity', 'stripclub'),
                    ('amenity', 'skateboard_parking'),
                    ('amenity', 'concert_hall')}
        ignores = {('name', '*school*')}

        return self.__matches(entry, patterns, ignores)

    def __is_company(self, entry: dict) -> bool:
        patterns = {('office', 'consulting'), ('office', 'company'),
                    ('office', 'insurance'), ('office', 'lawyer'),
                    ('company', '*'), ('name', '*kpmg*'),
                    ('office', 'estate_agent'), ('name', '*company*'),
                    ('office', 'laywer'), ('office', 'advertising_agency'),
                    ('office', 'exterminator'), ('studio', 'television'),
                    ('office', 'marketing'), ('office', 'television'),
                    ('name', 'spotify'), ('name', 'daum'),
                    ('office', 'it'), ('category', 'kitchen remodeling'),
                    ('studio', 'video'), ('name', '*insurance*'),
                    ('name', '*consulting*'), ('office', 'transport'),
                    ('name', 'mesa revenue partners'), ('name', '*kwkw-am*'),
                    ('name', 'stannard hall'), ('name', 'liton'),
                    ('name', 'we can foundation, inc'), ('name', '*studio*'),
                    ('name', 'hub on campus los angeles'),
                    ('office', 'employment_agency'), ('studio', 'audio'),
                    ('name', 'black lotus communications, operations center'),
                    ('name', 'Attack! Marketing'), ('name', '* inc'),
                    ('office', 'coworking'), ('name', 'jim henson'),
                    ('office', 'moving_company'), ('office', 'graphic_design'),
                    ('amenity', 'animal_boarding')}
        ignores = {('amenity', 'restaurant'), ('amenity', 'bank'),
                   ('shop', '*'), ('craft', 'brewery'), ('tourism', '*'),
                   ('amenity', 'cafe'), ('historic', 'memorial'),
                   ('gnis:feature_type', 'airport'), ('amenity', 'dentist'),
                   ('office', 'educational_institution'), ('leisure', 'dance'),
                   ('leisure', 'fitness_centre'), ('amenity', 'theatre'),
                   ('craft', 'pottery'), ('public_transport', 'station'),
                   ('name', '*fire station*'), ('leisure', 'sports_centre'),
                   ('amenity', 'post_office'), ('amenity', 'fast_food'),
                   ('operator', '*Metropolitan Transportation Authority*'),
                   ('name', 'studio city'), ('amenity', 'clinic'),
                   ('healthcare', 'counselling'), ('amenity', 'arts_centre'),
                   ('public_transport', 'platform')}

        return self.__matches(entry, patterns, ignores)

    def __is_organization(self, entry: dict) -> bool:
        patterns = {('office', 'foundation'), ('leisure', 'hackerspace'),
                    ('religion', '*'), ('name', 'saint agathas hall'),
                    ('name', 'melchizedek love & light healing center'),
                    ('name', 'ahmanson commons'), ('name', '*catholic*'),
                    ('amenity', 'community_centre'), ('name', 'kaos networks'),
                    ('name', '*community corporation*'), ('office', 'ngo'),
                    ('amenity', 'place_of_worship'), ('office', 'association'),
                    ('name', '*institute*'), ('name', '*guild of america'),
                    ('name', '*church*'), ('office', 'religion')}
        ignores = {('tourism', '*'), ('preschool', '*'), ('shop', '*'),
                   ('amenity', 'restaurant'), ('amenity', 'school'),
                   ('amenity', 'clinic'), ('amenity', 'college'),
                   ('social_facility', 'food_bank'), ('amenity', 'hospital'),
                   ('operator', '*Metropolitan Transportation Authority*'),
                   ('amenity', 'research_institute'), ('amenity', 'doctors'),
                   ('historic', 'wayside_shrine'), ('amenity', 'university'),
                   ('amenity', 'fast_food'), ('amenity', 'university'),
                   ('name', 'Jules Stein Eye Institute'),
                   ('name', 'Church & 405 Off-ramp')}

        return self.__matches(entry, patterns, ignores)

    def __is_safety(self, entry: dict) -> bool:
        patterns = {('emergency', '*'), ('enforcement', '*'),
                    ('amenity', 'fire_station'), ('amenity', 'police'),
                    ('amenity', 'ranger_station'), ('bunker_type', '*'),
                    ('military', '*'), ('government', 'defender'),
                    ('amenity', 'shelter'), ('amenity', 'courthouse'),
                    ('operator', 'los angeles police department'),
                    ('government', 'military_recruitment'),
                    ('surveillance:type', 'camera'), ('name', '*court*'),
                    ('surveillance', 'outdoor'), ('name', '*crime lab*'),
                    ('operator', 'los angeles county sheriff\'s department'),
                    ('name', 'law offices'), ('amenity', 'security_booth'),
                    ('name', '*fire station*')}
        ignores = {('amenity', 'cafe'), ('name', '*courtyard*'),
                   ('name', '*courtland*'), ('social_facility', 'food_bank'),
                   ('name', '*harcourt*'), ('name', '*courtleigh*'),
                   ('tourism', 'artwork'), ('amenity', 'hospital'),
                   ('amenity', 'fast_food'), ('amenity', 'food_court'),
                   ('gnis:feature_type', 'airport'), ('amenity', 'dentist'),
                   ('name', '*food court*'), ('tourism', 'gallery'),
                   ('shop', 'convenience'), ('office', 'estate_agent'),
                   ('leisure', 'pitch')}

        return self.__matches(entry, patterns, ignores)

    def __is_area(self, entry: dict) -> bool:
        patterns = {('place', 'quarter'), ('name', '*plaza*'),
                    ('name', 'little tokyo'), ('name', 'new chinatown'),
                    ('name', 'dogtown'), ('place', 'neighbourhood'),
                    ('place', 'suburb'), ('place', 'locality'),
                    ('name', 'Granada Building'), ('amenity', 'marketplace'),
                    ('place', 'square')}
        ignores = {('tourism', '*'), ('amenity', 'food_court'), ('shop', '*'),
                   ('name', '*hotel*'), ('amenity', 'post_office'),
                   ('amenity', 'dentist'), ('amenity', 'doctors'),
                   ('amenity', 'events_venue')}

        return self.__matches(entry, patterns, ignores)

    def __is_government(self, entry: dict) -> bool:
        patterns = {('diplomatic', 'consulate'), ('power', 'substation'),
                    ('operator:type', 'public'), ('consulate', '*'),
                    ('pumping_station', '*'), ('townhall:type', 'city'),
                    ('government', 'public_service'), ('*', 'district office'),
                    ('amenity', 'townhall'), ('name', '*department*'),
                    ('office', 'government')}
        ignores = {('operator', '*sheriff*'), ('operator', '*police*'),
                   ('amenity', 'parking'), ('government', 'defender'),
                   ('amenity', 'library'), ('amenity', 'school'),
                   ('amenity', 'bicycle_rental'), ('amenity', 'hospital'),
                   ('military', 'office'), ('name', '*mental health*'),
                   ('amenity', 'clinic'), ('office', 'research'),
                   ('government', 'military_recruitment'),
                   ('amenity', 'post_office')}

        return self.__matches(entry, patterns, ignores)

    def __is_public(self, entry: dict) -> bool:
        patterns = {('amenity', 'waste_basket'), ('water', 'reservoir'),
                    ('waterway', 'weir'), ('amenity', 'toilets'),
                    ('amenity', 'telephone'), ('amenity', 'parcel_locker'),
                    ('maxspeed', '*'), ('social_facility', '*'),
                    ('tower:type', '*'), ('backrest', '*'), ('fountain', '*'),
                    ('amenity', 'drinking_water'), ('leisure', 'garden'),
                    ('leisure', 'dog_park'), ('leisure', 'park'),
                    ('amenity', 'library'), ('amenity', 'social_facility'),
                    ('crossing:barrier', '*'), ('leisure', 'picnic_table'),
                    ('amenity', 'bench'), ('waterway', 'dam'),
                    ('amenity', 'public_bookcase'), ('amenity', 'fountain'),
                    ('natural', 'beach')}
        ignores = {('name', '*church*'), ('operator', '*sheriff*'),
                   ('name', '*plaza*'), ('name', '*catholic*'),
                   ('social_facility', 'nursing_home'), ('sport', 'chess'),
                   ('name', '*institute*'), ('military', 'office'),
                   ('historic', 'monument'), ('emergency', 'siren'),
                   ('amenity', 'shelter')}

        return self.__matches(entry, patterns, ignores)

    def __is_other(self, entry: dict) -> bool:
        patterns = {('name', 'netflix on vine campus'),
                    ('landuse', 'residential'), ('name', 'the row house'),
                    ('name', '*')}
        ignores = {('name', '*hotel*'), ('name', '*company*'),
                   ('name', '*plaza*')}

        return self.__matches(entry, patterns, ignores)

    def __matches(self, poi: dict, refs: set, ignore: set = None) -> bool:
        if ignore is not None and self.__search(ignore, poi):
            return False
        return self.__search(refs, poi)

    def __search(self, search_patterns: set, poi: dict) -> bool:
        poi = {key.lower(): value for key, value in poi.items()}
        property_keys = list({key for key, _ in poi.items()})
        for key, value in search_patterns:
            key_matches = self.__contains(key, property_keys)
            if not key_matches:
                continue
            property_values = [poi[match].lower() for match in key_matches]
            value_matches = self.__contains(value, property_values)

            if any(value_matches):
                return True

        return False

    @staticmethod
    def __contains(keyword: str, elements: list) -> list:
        if keyword.startswith('*') and keyword.endswith('*'):
            keyword = keyword[1:-1].lower() if len(keyword) > 1 else ''
            matches = [keyword in ele for ele in elements]
        elif keyword.startswith('*'):
            keyword = keyword[1:].lower() if len(keyword) > 1 else ''
            matches = [ele.endswith(keyword) for ele in elements]
        elif keyword.endswith('*'):
            keyword = keyword[:-1].lower() if len(keyword) > 1 else ''
            matches = [ele.startswith(keyword) for ele in elements]
        else:
            keyword = keyword.lower()
            matches = [ele == keyword for ele in elements]

        return [value for value, match in zip(elements, matches) if match]

    def __normalize(self):
        for _, entry in self.data['poi'].items():
            properties = entry['properties']

            # lowercase value
            for property_key, value in properties.items():
                if isinstance(value, str):
                    properties[property_key] = value.lower()

            # lowercase key
            entry['properties'] = {key.lower(): value
                                   for key, value in properties.items()}

            # convert to integer value
            if 'addr:postcode' in properties:
                value_tmp = utils.normalize_int(properties['addr:postcode'])
                entry['properties']['addr:postcode'] = value_tmp

    def __repair(self):
        for _, entry in self.data['poi'].items():
            properties = entry['properties']

            if 'country' in properties and 'addr:country' not in properties:
                entry['properties']['addr:country'] = properties['country']

    def __cluster_properties(self):
        for _, entry in self.data['poi'].items():
            properties = entry['properties']

            fields = {'art_name': ['old_name', 'name_1', 'alt_name_1',
                                   'name:en', 'short_name:en', 'short_name',
                                   'official_name:en', 'alt_name', 'name2',
                                   'alt_name:en', 'stop_name', 'name',
                                   'official_name', 'flag:name', 'subject',
                                   'addr:housename', 'company'],
                      'art_phone': ['phone', 'phone_1', 'contact:phone',
                                    'emergency'],
                      'art_operator': ['operator', 'owner', 'operator:short',
                                       'operator:type'],
                      'art_description': ['artist_name', 'tourism', 'baseball',
                                          'camera:mount', 'camera:type',
                                          'source', 'siren:type', 'memorial',
                                          'board_type', 'fitness_station',
                                          'military', 'office', 'shelter',
                                          'craft', 'dance:teaching', 'is_in',
                                          'memorial:type', 'sport', 'access',
                                          'surveillance:type', 'school_type',
                                          'training:for', 'support', 'network',
                                          'consulate', 'natural', 'historic',
                                          'manufacturer', 'source:feature',
                                          'healthcare', 'historic:amenity',
                                          'tower:type', 'siren:model', 'level',
                                          'dispensing', 'shop', 'beauty',
                                          'cuisine', 'healthcare:speciality',
                                          'siren:purpose', 'brand', 'leisure',
                                          'religion', 'advertising', 'place',
                                          'social_facility', 'branch', 'club',
                                          'townhall:type', 'self_service',
                                          'social_facility:for', 'highway',
                                          'denomination', 'category', 'target',
                                          'diplomatic', 'public_transport',
                                          'amenity', 'clothes', 'material',
                                          'description', 'information', 'note',
                                          'shelter_type', 'surveillance',
                                          'landuse', 'artwork_type', 'screen',
                                          'fire_hydrant:position', 'playground',
                                          'education_profile', 'studio',
                                          'community_centre', 'water_source',
                                          'community_centre:for', 'colour',
                                          'theatre:type', 'fire_hydrant:type'],
                      'art_opening_hours': ['opening_hours', 'service_times',
                                            'opening_hours:kitchen',
                                            'opening_hours:covid19',
                                            'opening_hours:pharmacy'],
                      'art_link': ['url', 'facebook', 'twitter', 'youtube',
                                   'opening_hours:url', 'network:wikipedia',
                                   'flag:wikidata', 'owner:wikipedia',
                                   'brand:wikipedia', 'contact:facebook',
                                   'contact:instagram', 'contact:website',
                                   'contact:linkedin', 'website:menu',
                                   'contact:twitter', 'network:wikidata',
                                   'operator:wikidata', 'brand:wikidata',
                                   'wikidata', 'wikipedia', 'website',
                                   'operator:wikipedia', 'brand:website',
                                   'subject:wikidata']}

            for key, values in fields.items():
                new_field = {properties[value]
                             for value in values
                             if value in properties
                             if properties[value] not in ['yes', 'no']}

                if key == 'art_description':
                    new_field = {value
                                 for entry in new_field
                                 for value in entry.split(';')}
                if key == 'art_link':
                    new_field = {'https://en.wikipedia.org/wiki/' + value
                                 if value.startswith('en:') else value
                                 for value in new_field}

                properties[key] = list(new_field)

            art_provides = {property_key
                            for property_key, value in properties.items()
                            if value == 'yes'}
            properties['art_provides'] = list(art_provides)

    def to_schema(self, schema: Schema) -> dict:
        pid_to_data = {}
        for category_key, pid_list in tqdm(self.data['category_to_pid'].items(), bar_format=BAR_FORMAT):
            for pid in pid_list:
                tmp = self.data['poi'][pid]
                properties = tmp['properties']

                entry = {'sid': pid,
                         'source': self.identifier,
                         'category': category_key,
                         'location': {
                             'latitude': tmp['geometry']['coordinates'][1],
                             'longitude': tmp['geometry']['coordinates'][0],
                             'street': properties.get('addr:street', None),
                             'house_number': properties.get('addr:housenumber', None),
                             'city': properties.get('addr:city', None),
                             'postcode': properties.get('addr:postcode', None),
                             'state': properties.get('addr:state', None),
                             'country': properties.get('addr:country', None)},
                         'properties': {
                             'name': properties['art_name'],
                             'phone': properties['art_phone'],
                             'operator': properties['art_operator'],
                             'description': properties['art_description'],
                             'opening_hours': properties['art_opening_hours'],
                             'provides': properties['art_provides'],
                             'link': properties['art_link']}}

                try:
                    validated = schema.validate(entry)
                except SchemaError:
                    log.error('> POI schema not applicable: %s', pid)
                    continue

                pid_to_data[pid] = validated

        return pid_to_data
