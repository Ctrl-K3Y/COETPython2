import matplotlib.pyplot as plt
from earthquakes import *
from pathlib import *
import numpy as np
import json
import sys
"""
Filename: earthquakes_analyser.py
Assignment: COET 295 Assignment 1
Author: Kizza Alba (alba0877)
Instructors: Wade Lahoda, Bryce Barrie
Date: 2024-05-14
"""


def start_menu(quake_data):
    """ Main menu function used to display earthquake data menu to the user"""
    while True:
        selection = input("\n\t\t\t----[Menu]---- \n (1) Set location filter \t (5) Display exceptional quakes"
                          " \n (2) Set property filter \t (6) Display magnitude stats "
                          "\n (3) Clear filters \t\t (7) Plot quake map "
                          "\n (4) Display Quakes \t\t (8) Plot magnitude stats \n (9) Quit \n\n Choose Option: ")
        selection = selection.strip(" \t\n\r")
        if selection == "1":
            prompt_location_input(quake_data)
        elif selection == "2":
            prompt_property_input(quake_data)
        elif selection == "3":
            quake_data.clear_filters()
        elif selection == "4":
            print("\n\t\t----[Displaying Earthquakes]----")
            display_quake_data(quake_data.get_filtered_array())
        elif selection == "5":
            print("\n\t\t----[Displaying Exceptional Earthquakes]----")
            display_exceptional_quakes(quake_data.get_filtered_array())
        elif selection == "6":
            print("\n\t\t----[Displaying Magnitude Statistics]----")
            display_magnitude_stats(quake_data.get_filtered_array())
        elif selection == "7":
            print("\n\t\t----[Loading Earthquake Scatter Map]----")
            plot_quake_map(quake_data.get_filtered_array())
        elif selection == "8":
            print("\n\t\t----[Loading Earthquake Magnitude Chart]----")
            plot_magnitude_chart(quake_data.get_filtered_array())
        elif selection == "9":
            print("\nThank you for using Earthquake Analyser! Goodbye!")
            break
        else:
            print("\n[!] Invalid selection. Please try again [!]")


def plot_magnitude_chart(quake_array):
    """Creates a chart that shows the number of quakes with whole number magnitudes"""
    whole_numbered_mags = (quake_array[((quake_array['magnitude'] % 1) == 0)])['magnitude']
    categories = np.unique(whole_numbered_mags)
    values = [np.sum(whole_numbered_mags == value) for value in categories]
    plt.bar(categories,values,color='c')
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Quakes')
    plt.title('Magnitude Chart')
    plt.show()


def plot_quake_map(quake_array):
    """ Create a scatter map which shows where """
    plt.figure(figsize=(10, 6))
    plt.scatter(quake_array['lat'], quake_array['long'], np.power(quake_array['magnitude'],2),c='r')
    plt.xlabel('Latitude')
    plt.xlabel('Longitude')
    plt.title("Scatter map of earthquakes")

    plt.grid(True)
    plt.show()


def display_magnitude_stats(quake_array):
    """ Calculates the mean, median, mode, and standard array of the quake data """
    mean_of_magnitude = np.mean(quake_array['magnitude'])
    median_of_magnitude = np.median(quake_array['magnitude'])
    std_of_magnitude = np.std(quake_array['magnitude'])

    rounded_down_values = np.floor(quake_array['magnitude'])
    occurrences_modes = {value: np.sum(rounded_down_values == value) for value in np.unique(rounded_down_values)}
    mode_of_magnitude = max(occurrences_modes, key=occurrences_modes.get)
    print(f"Mean: {mean_of_magnitude},  Mode: {mode_of_magnitude}"
          f", Median: {median_of_magnitude}, Standard Deviation: {std_of_magnitude}")


def display_exceptional_quakes(quake_array):
    """
        Determines all the quakes with magnitudes that are
        above one standard deviation of the data's mean
    """
    if len(quake_array) == 0:
        print("[!] Cannot perform calculations on an empty array [!]")
        return

    mean_of_magnitudes = np.mean(quake_array['magnitude'])
    std_of_magnitudes = np.std(quake_array['magnitude'])
    upper = mean_of_magnitudes + std_of_magnitudes
    quakes_above_one_std_deviation_above_mean = quake_array[(quake_array['magnitude'] > upper)]
    display_quake_data(quakes_above_one_std_deviation_above_mean)


def display_quake_data(quake_array):
    """
        Goes through each quakes in the quake array converting them
        to objects and displaying them
    """
    if quake_array.size == 0:
        print("[!] Nothing to display, Filtered data is empty [!]")
        return
    for quake in quake_array:
        print(quake['quake'].__str__())




def prompt_location_input(quake_data):
    """
        Prompts the user for latitude, longitude, and distance criteria to
        apply as location filter to apply to the existing quake data
    """
    latitude = input("Please supply a latitude: ")
    longitude = input("Please supply a longitude: ")
    distance = input("Please supply a distance to compare: ")
    try:
        quake_data.set_location_filter(float(latitude), float(longitude), float(distance))
    except ValueError as err:
        print(f"[ValueError]: {err}")


def prompt_property_input(quake_data):
    """
        Prompts the user for a magnitude, felt, and significance criteria
        to apply as property filter to the existing
    """
    print("Please supply a minimum of one argument for this operation")
    magnitude = input("Please supply the magnitude: ")
    felt = input("Please supply the felt: ")
    significance = input("Please supply the significance: ")
    # A tuple that only includes inputs with valid data
    arguments = (float(magnitude) if len(magnitude) != 0 else None,
                 int(felt) if len(felt) != 0 else None,
                 int(significance) if len(significance) != 0 else None)

    if all(arg is None for arg in list(arguments)):
        print("!WARNING: A single argument is required for this operation, you provided 0")

    quake_data.set_property_filter(*arguments)


def read_json(filepath="earthquakes.geojson"):
    """
        Reads data from the given file if it exists else an error will be raised
    """
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

print("----------------Welcome to Earthquake Analyser!----------------")
print("Please supply a filename to analyze (optional): ")

filename = sys.stdin.readline().strip()
print(f"FILE CHOSEN: {filename}")
geojson = read_json(filename) if len(filename) != 0 else read_json()
quake_data = QuakeData(geojson)

start_menu(quake_data)
