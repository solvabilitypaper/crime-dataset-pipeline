"""*
 *     Crime Dataset Pipeline
 *
 *        File: utils.py
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

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import gc
import logging as log
import time
import sys

import geopy.distance
import pandas as pd
from pandas import Series
import requests
from shapely.geometry import shape, Point
from shapely.errors import GEOSException

from constants import NEARBY_RADII


def normalize_column_date(column: Series, regex: str) -> Series:
    column = pd.to_datetime(column, format=regex)
    column = column.map(pd.Timestamp.timestamp)

    return column


def normalize_column_int(column: Series, none_dummy: int = -100) -> Series:
    column = column.fillna(none_dummy).astype('int32')
    column = column.replace(none_dummy, None)

    return column


def normalize_column_str(column: Series) -> Series:
    column = column.astype(str).str.lower()
    column = column.replace(r'\s+', ' ', regex=True)
    column = column.replace('none', None)

    return column


def normalize_int(value: str) -> int or None:
    try:
        return int(value)
    except ValueError:
        return None


def normalize_float(value: str) -> float or None:
    try:
        return float(value)
    except ValueError:
        return None


def normalize_str(value: str) -> str or None:
    value = value.lower()
    value = value.replace(r'\s+', ' ')
    value = value if 'none' != value else None

    return value


def handle_request(target_url: str, params: dict) -> requests.Response or None:
    try:
        reverse_response = requests.get(target_url, params=params, timeout=120)
    except requests.exceptions.Timeout:
        log.info('>> Connection timeout. Pausing for 120 seconds.')
        time.sleep(120)
        return None
    except requests.exceptions.ConnectionError:
        log.info('>> Connection error. Pausing for 300 seconds.')
        time.sleep(300)
        return None

    http_code = reverse_response.status_code
    if http_code != 200:
        log.debug('>> Status code %s. Pausing for 60 seconds.', http_code)
        time.sleep(60)
        return None

    return reverse_response


def is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def contains(latitude: float, longitude: float, coordinates: dict) -> bool:
    point = Point(longitude, latitude)
    # TODO point needs to be the same order as coordinates
    polygon = shape(coordinates)

    try:
        return polygon.contains(point)
    except GEOSException as error_msg:
        log.error(error_msg)
        log.error('Coordinates: %s', coordinates)
        log.error('Latitude: %s | Longitude: %s', latitude, longitude)
        return False


def _get_point_distance(coordinates: list) -> list:
    result = []

    for crime_key, poi_key, crime_point, poi_point in coordinates:
        distance = geopy.distance.geodesic(crime_point, poi_point)
        distance = getattr(distance, 'm')

        if distance <= NEARBY_RADII[-1]:
            result.append((crime_key, poi_key, distance))

    return result


def get_distances(coordinate_batches: list, num_workers: int) -> list:
    start = time.time()

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    num_pairs = 0
    for batch in coordinate_batches:
        num_pairs += len(batch)
        futures.append(executor.submit(_get_point_distance, batch))

    log.info('> Started %s worker processes', len(futures))
    log.info('>> They process %s coordination pairs', num_pairs)
    log.info('>> Waiting for the worker processes to finish...')
    wait(futures, return_when=ALL_COMPLETED)

    result = [entry for future in futures for entry in future.result()]
    log.info('>> Completed (Duration: %.2f seconds)', (time.time() - start))

    executor.shutdown()
    futures.clear()
    gc.collect()  # just to be sure gc is called

    return result


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log.error('Uncaught exception',
              exc_info=(exc_type, exc_value, exc_traceback))
