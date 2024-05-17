import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from earthquakes import *
from pathlib import *
from nphelpers import printnp


def start_menu():
    """Main menu function used to display earthquake data menu"""
    print("Welcome to Earthquake Analyzer!")
    filename = input("Please supply a filename to analyze: ")
    geojson = read_json(filename) if len(filename) != 0 else read_json()

    quake_data = QuakeData(geojson)
    while True:
        selection = input("\t\t\t\t\t----[Menu]---- \n (1) Set location filter \t (5) Display exceptional quakes"
                          " \n (2) Set property filter \t (6) Display magnitude stats "
                          "\n (3) Clear filters \t\t\t (7) Plot quake map "
                          "\n (4) Display Quakes \t\t (8) Plot magnitude stats \n (9) Quit \n\n Choose Option: ")
        if selection == "1":
            prompt_location_input(quake_data)
            print("\t\t\t----[Location filter has been applied]----")
        elif selection == "2":
            prompt_property_input(quake_data)
            print("\t\t\t----[Property filter has been applied]----")
        elif selection == "3":
            quake_data.clear_filters()
            print("\t\t\t----[Filters Have Been Cleared]----")
        elif selection == "4":
            print("\t\t\t----[Displaying Earthquakes]----")
            display_quake_data(quake_data.get_filtered_array())
        elif selection == "5":
            print("\t\t----[Displaying Exceptional Earthquakes]----")
            display_exceptional_quakes(quake_data.get_filtered_array())
        elif selection == "6":
            print("\t\t\t----[Displaying Magnitude Statistics]----")
            display_magnitude_stats(quake_data.get_filtered_array())
        elif selection == "7":
            print("\t\t----[Loading Earthquake Scatter Map]----")
            display_quake_map(quake_data.get_filtered_array())
        elif selection == "8":
            print("\t\t\t----[Loading Earthquake Magnitude Chart]----")
            display_magnitude_chart(quake_data.get_filtered_array())
        elif selection == "9":
            print("Thank you for using Earthquake Analyzer! Goodbye!")
            break


def display_magnitude_chart(quake_array):
    """Creates a chart that shows the number of quakes with whole number magnitudes"""
    whole_numbered_mags = (quake_array[((quake_array['magnitude'] % 1) == 0)])['magnitude']
    categories = np.unique(whole_numbered_mags)
    values = [np.sum(whole_numbered_mags == value) for value in categories]
    plt.bar(categories,values)
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Quakes')
    plt.title('Magnitude Chart')
    plt.show()


def display_quake_map(quake_array):

    plt.figure(figsize=(10, 6))
    plt.scatter(quake_array['lat'], quake_array['long'], quake_array['magnitude'] * 5, c='b')
    plt.xlabel('Latitude')
    plt.xlabel('Longitude')
    plt.title("Scatter map of earthquakes")

    plt.grid(True)
    plt.show()


def display_magnitude_stats(quake_array):
    mean_of_magnitude = np.mean(quake_array['magnitude'])
    median_of_magnitude = np.median(quake_array['magnitude'])
    std_of_magnitude = np.std(quake_array['magnitude'])

    rounded_down_values = np.floor(quake_array['magnitude'])
    occurrences_modes = {value: np.sum(rounded_down_values == value) for value in np.unique(rounded_down_values)}
    mode_of_magnitude = max(occurrences_modes, key=occurrences_modes.get)
    print(f"Mean: {mean_of_magnitude},  Mode: {mode_of_magnitude}"
          f", Median: {median_of_magnitude}, Standard Deviation: {std_of_magnitude}")


def display_exceptional_quakes(quake_array):

    if len(quake_array) == 0:
        print("ERROR: Cannot perform calculations on an empty array")
        return

    mean_of_magnitudes = np.mean(quake_array['magnitude'])
    std_of_magnitudes = np.std(quake_array['magnitude'])
    upper = mean_of_magnitudes + std_of_magnitudes
    quakes_above_one_std_deviation_above_mean = quake_array[(quake_array['magnitude'] > upper)]
    display_quake_data(quakes_above_one_std_deviation_above_mean)


def display_quake_data(quake_array):
    quake_objects = []
    if quake_array.size == 0:
        print("The filtered array is empty")
    else:
        for quake in quake_array:
            quake_objects.append(data_to_quake_object(quake))
        for obj in quake_objects:
            print(obj.__str__())


def data_to_quake_object(quake):
    current_quake = quake['quake']
    time = current_quake['properties']['time']
    q_type = current_quake['properties']['type']
    coords = (quake['lat'], quake['long'])
    return Quake(quake['magnitude'], time, quake['felt'], quake['significance'], q_type, coords)


def prompt_location_input(quake_data):
    latitude = input("Please supply a latitude: ")
    longitude = input("Please supply a longitude: ")
    distance = input("Please supply a distance to compare: ")

    quake_data.set_location_filter(float(latitude), float(longitude), float(distance))


def prompt_property_input(quake_data):

    print("Please supply a minimum of one argument for this operation")
    magnitude = input("Please supply the magnitude: ")
    felt = input("Please supply the felt: ")
    significance = input("Please supply the significance: ")
    arguments = (float(magnitude) if len(magnitude) != 0 else None,
                 int(felt) if len(felt) != 0 else None,
                 int(significance) if len(significance) != 0 else None)

    if all(arg is None for arg in list(arguments)):
        print("WARNING: A single argument is required for this operation, you provided 0")

    quake_data.set_property_filter(*arguments)


def read_json(filepath="earthquakes.geojson"):
    filepath = Path(f"../Data/{filepath}")
    geojson = {}
    if filepath.exists():
        try:
            geojson = json.loads(filepath.read_text())
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        raise FileExistsError("File supplied does not exist!")
    return geojson


# vectorized_data_display = np.vectorize(display_quake_data)
# vectorized_mag_stats_display = np.vectorize(display_magnitude_stats)
start_menu()
# geojson = read_json("../Data/earthquakes.geojson")
# qd = QuakeData(geojson)
# # qd.set_property_filter(felt=2, significance=285)
# # qd.set_location_filter(-115.2061667,32.4033333,100000000)
# # display_quake_map(qd.get_filtered_array())
# display_magnitude_chart(qd.get_filtered_array())
# # display_exceptional_quakes(qd)
# # display_magnitude_stats(qd.get_filtered_array())
