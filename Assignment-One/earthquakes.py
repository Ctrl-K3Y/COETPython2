# Filename: earthquakes.py
# Assignment: COET 295 Assignment 1
# Author: Kizza Alba (alba0877)
# Instructors: Wade Lahoda, Bryce Barrie
# Date: 2024-05-14
import math
from nphelpers import printnp
import numpy as np
from pathlib import *
import json


# Purpose: Uses the haversine law to calculate distance between two coordinates.
# Arguments: latitude and longitude of two coordinates
def calc_distance(lat1, long1, lat2, long2):
    """Calculates the distance between two points"""
    earth_radius = 6371e3
    lat1 = float(lat1)
    lat2 = float(lat2)
    long1 = float(long1)
    long2 = float(long2)

    first_lat_rad = lat1 * math.pi / 180
    second_lat_rad = lat2 * math.pi / 180
    delta_latitude_rad = (lat2 - lat1) * math.pi / 180
    delta_longitude_rad = (long2 - long1) * math.pi / 180

    square_of_half_chord = math.sin(delta_latitude_rad / 2) * math.sin(delta_latitude_rad / 2) + \
                           math.cos(first_lat_rad) * math.cos(second_lat_rad) * \
                           math.sin(delta_longitude_rad / 2) * math.sin(delta_longitude_rad)
    angular_distance_rad = 2 * math.atan2(math.sqrt(square_of_half_chord), math.sqrt(1 - square_of_half_chord))

    return (earth_radius * angular_distance_rad) * 1000


def convert_json_to_object(filepath="../Data/earthquakes.geojson"):
    filepath = Path(filepath)
    if filepath.exists():
        geojson = json.loads(filepath.read_text())
    else:
        geojson = {}
    return geojson


# convert_json_to_object("../Data/earthquakes.geojson") - REMOVE

# A constructor(geojson) should take in a Python dictionary of geojson data
#   ▪ The constructor should go through the dictionary retrieve a
#   list of earthquakes from the ‘features’ entry in the dictionary
#   ▪ It should discard the data for any feature where:
#   • The type is not “feature”
#   • The ‘properties’ sub-dictionary is missing any of the following keys:
#               ‘mag’,’time’,felt’,’sig’,’type’,’magType’s
#   • The ‘geometry’ sub-dictionary does not contain a ‘type’ key with the value ‘Point’
#   • The ‘geometry’ sub-dictionary does not contain a ‘coordinates’ key whose value is a
#       tuple of three numbers
#   ▪ The rest of the quakes should be stored in the quake_array attribute
#   ▪ You may assume after the constructor is run no further data will be added to the object,
#       though filters may be applied

#  An attribute called quake_array which contains a structured numpy array with at least the
# following fields: quake(object), magnitude(float), felt(int32), significance(int32), lat(float),
# long(float)

class QuakeData:

    def __init__(self, geojson):
        included_features = []
        if 'features' in geojson:
            keys_to_check = ['mag', 'felt', 'sig', 'type', 'magType', 'time']
            keys_to_include = list(keys_to_check)[:-3]
            for feature in geojson['features']:
                if feature['type'] == 'Feature':
                    if all(key in feature['properties'] and \
                           feature['properties'][key] is not None for key in keys_to_check):
                        coordinates = list(feature['geometry']['coordinates'])
                        if feature['geometry']['type'] == 'Point' and len(coordinates) == 3:
                            earthquake_data = [feature]
                            earthquake_data = [feature['properties'][key] for key \
                                            in keys_to_include if key in feature['properties']]
                            earthquake_data += [coordinates[0], coordinates[1]]
                            included_features.append(tuple(earthquake_data))
        print(included_features)
        self.quake_array = np.array(included_features, dtype=[
            # ('quake', 'O'),
            ('magnitude', 'float64'),
            ('felt', 'int32'),
            ('significance', 'int32'),
            ('lat', 'float64'),
            ('long', 'float64')
        ])


QuakeData(convert_json_to_object())
