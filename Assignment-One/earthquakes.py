# Filename: earthquakes.py
# Assignment: COET 295 Assignment 1
# Author: Kizza Alba (alba0877)
# Instructors: Wade Lahoda, Bryce Barrie
# Date: 2024-05-14
import math
from nphelpers import printnp
import numpy as np
import copy
from pathlib import *
import json


# Purpose: Uses the haversine law to calculate distance between two coordinates.
# Arguments: latitude and longitude of two coordinates
def calc_distance(lat1, long1, lat2, long2):
    """Calculates the distance between two points"""
    earth_radius = 6371e3

    first_lat_rad = lat1 * math.pi / 180
    second_lat_rad = lat2 * math.pi / 180
    delta_latitude_rad = (lat2 - lat1) * math.pi / 180
    delta_longitude_rad = (long2 - long1) * math.pi / 180

    square_of_half_chord = math.sin(delta_latitude_rad / 2) * math.sin(delta_latitude_rad / 2) + \
                           math.cos(first_lat_rad) * math.cos(second_lat_rad) * \
                           math.sin(delta_longitude_rad / 2) * math.sin(delta_longitude_rad)
    angular_distance_rad = 2 * math.atan2(math.sqrt(square_of_half_chord), math.sqrt(1 - square_of_half_chord))

    return (earth_radius * angular_distance_rad) * 1000





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
                            earthquake_data = [feature, coordinates[0], coordinates[1]]
                            earthquake_data += [feature['properties'][key] for key \
                                                in keys_to_include if key in feature['properties']]
                            included_features.append(tuple(earthquake_data))
        self.quake_array = np.array(included_features, dtype=[
            ('quake', 'O'),
            ('lat', 'float64'),
            ('long', 'float64'),
            ('magnitude', 'float64'),
            ('felt', 'int32'),
            ('significance', 'int32'),
        ])
        self.vectorized_distance_calculation = np.vectorize(calc_distance)
        self.location_filter = self.quake_array is not None
        self.property_filter = self.quake_array is not None
        self.filtered_array = np.copy(self.quake_array)

    def set_location_filter(self, latitude, longitude, distance):
        quake_filter = self.vectorized_distance_calculation(self.quake_array['lat'], self.quake_array['long'], latitude,
                                                            longitude) <= distance
        self.location_filter = quake_filter
        print("Location filter set")

    def set_property_filter(self, magnitude=None, felt=None, significance=None):
        relevant_parameters = []
        property_set_status = ""

        try:
            relevant_parameters = {args: arg_value for args, arg_value in \
                                   zip(['magnitude', 'felt', 'significance'], [magnitude, felt, significance]) if
                                   arg_value is not None}  # TODO: Document what zip does
            if len(relevant_parameters) == 0:
                raise ValueError("Property could not be set, at least one parameter must be specified")
        except ValueError as e:
            print(f"[ValueError]: {e}")
            return
        print(relevant_parameters)
        quake_filter = self.quake_array is not None
        for key, value in relevant_parameters.items():
            quake_filter &= (self.quake_array[key] == value)
        self.property_filter = quake_filter

    def clear_filters(self):
        self.filtered_array = np.copy(self.quake_array)
        self.location_filter = self.quake_array is not None
        self.property_filter = self.quake_array is not None

    def get_filtered_array(self):
        self.filtered_array = self.quake_array[(self.property_filter & self.location_filter)]
        if self.filtered_array.size > 0 and isinstance(self.filtered_array[0], np.ndarray):
            self.filtered_array = self.filtered_array[0]
        return self.filtered_array

    def get_filtered_list(self):
        return self.get_filtered_array().tolist()


class Quake:
    def __init__(self, magnitude, time, felt, significance, q_type, coords):
        self.magnitude = magnitude
        self.time = time
        self.felt = felt
        self.significance = significance
        self.q_type = q_type
        self.coords = coords
        self.vectorized_distance_calculation = np.vectorize(calc_distance)

    def __str__(self):
        return f"{self.magnitude} Magnitude Earthquake, {self.significance} Significance, felt by {self.felt} people in {self.coords[0]}, {self.coords[0]}"

    def get_distance_from(self, latitude, longitude):
        return self.vectorized_distance_calculation(self.coords[0], self.coords[1], latitude,
                                                    longitude)


# qd = QuakeData(convert_json_to_object())
# qd.set_location_filter(-90.9175, 13.8163, 100000000)
# qd.set_property_filter(felt=2, significance=285)
#
# printnp(qd.get_filtered_array())

# -77.5153333333333, 37.7101666666667, 2000
#-90.9175, 13.8163,
# qd.set_location_filter(-90.9175, 13.8163, 100000000)
