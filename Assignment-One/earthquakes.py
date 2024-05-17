import math
import numpy as np

"""
Filename: earthquakes.py
Assignment: COET 295 Assignment 1
Author: Kizza Alba (alba0877)
Instructors: Wade Lahoda, Bryce Barrie
Date: 2024-05-14
"""


def calc_distance(lat1, long1, lat2, long2):
    """Calculates the distance between two points using haversine law"""
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


class QuakeData:
    """A class used to represent earthquake information extracted from a file"""

    def __init__(self, geojson):
        """Constructor for QuakeData to store valid data into a structured array"""
        included_features = []
        if 'features' in geojson:
            keys_to_check = ['mag', 'felt', 'sig', 'type', 'magType', 'time']
            keys_to_include = list(keys_to_check)[:-3]
            for feature in geojson['features']:
                if feature['type'] == 'Feature':

                    # Processes all quake data that have valid attributes specified in the keys_to_check list
                    # all() - processes only the objects that meet all its conditions
                    if all(key in feature['properties'] and
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
        self.filtered_array = []

    def set_location_filter(self, latitude, longitude, distance):
        """
            Sets a location filter to narrow earthquake data to only the earthquakes within the specified
            distance of the coordinates
        """
        try:
            relevant_parameters = {float(arg_value) for arg_value in (latitude, longitude, distance) if
                                   arg_value is not None}
            if len(relevant_parameters) != 3:
                raise ValueError("Location filter ")
        except ValueError as e:
            print(f"[ValueError]: {e}")
            return
        quake_filter = self.vectorized_distance_calculation(self.quake_array['lat'],
                                                            self.quake_array['long'], float(latitude),
                                                            float(longitude)) <= float(distance)
        self.location_filter = quake_filter
        print("\n\t\t\t----[Location filter has been applied]----")

    def set_property_filter(self, magnitude=None, felt=None, significance=None):
        """
            Sets a property filter that narrows earthquake data to only the earthquakes
            that meet the specified criteria of magnitude, felt, and significance
        """
        try:
            # zip() - combines two lists of the same size as key-value pairs in a dictionary
            relevant_parameters = {args: arg_value for args, arg_value in \
                                   zip(['magnitude', 'felt', 'significance'], [magnitude, felt, significance]) if
                                   arg_value is not None}
            if len(relevant_parameters) == 0:
                raise ValueError("Property could not be set, at least one parameter must be specified")
        except ValueError as e:
            print(f"[ValueError]: {e}")
            return

        quake_filter = self.quake_array is not None

        # Appends to existing property filter based on the valid parameters given
        for key, value in relevant_parameters.items():
            quake_filter &= (self.quake_array[key] == value)
        self.property_filter = quake_filter
        print("\n\t\t----[Property filter has been applied]----")

    def clear_filters(self):
        """
            Removes the location and property filters and reverts filtered array to
            its original contents
        """
        self.filtered_array = np.copy(self.quake_array)
        self.location_filter = self.quake_array is not None
        self.property_filter = self.quake_array is not None
        print("\n\t\t\t----[Filters Have Been Cleared]----")

    def get_filtered_array(self):
        """
            Applies location and property filters to quake data
        :return: filtered array : numpy.ndarray
        """
        self.filtered_array = self.quake_array[(self.property_filter & self.location_filter)]
        if self.filtered_array.size > 0 and isinstance(self.filtered_array[0], np.ndarray):
            self.filtered_array = self.filtered_array[0]
        return self.filtered_array

    def get_filtered_list(self):
        """
            Converts filtered_array to a list
        :return: filtered list : list
        """
        return self.get_filtered_array().tolist()


class Quake:
    """ A class used to represent a single instance of an earthquake data   """

    def __init__(self, magnitude, time, felt, significance, q_type, coords):
        """Constructor for Quake class"""
        self.magnitude = magnitude
        self.time = time
        self.felt = felt
        self.significance = significance
        self.q_type = q_type
        self.coords = coords
        self.vectorized_distance_calculation = np.vectorize(calc_distance)

    def __str__(self):
        """ Returns a string formatted with quake data """
        return f"{self.magnitude} Magnitude Earthquake, {self.significance} Significance, felt by {self.felt} people in {self.coords[0]}, {self.coords[0]}"

    def get_distance_from(self, latitude, longitude):
        """ Returns the distance of quake from the coordinates specified """
        return self.vectorized_distance_calculation(self.coords[0], self.coords[1], latitude,
                                                    longitude)
