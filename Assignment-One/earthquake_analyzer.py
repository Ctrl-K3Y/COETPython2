from earthquakes import *
from pathlib import *


def start_menu():
    """Main menu function used to display earthquake data menu"""
    print("Welcome to Earthquake Analyzer!")
    filename = input("Please supply a filename to analyze: ")
    geojson = read_json(f"../Data/{filename}")

    quake_data = QuakeData(geojson)
    while True:
        selection = input("Please select an option: \n (1) Set location filter \n (2) Set property filter"
                          " \n (3) Clear filters \n (4) Display Quakes \n (5) Display exceptional quakes "
                          "\n (6) Display magnitude stats \n (7) Plot quake map \n (8) Plot magnitude stats \n (9) Quit")
        if selection == "1":
            prompt_location_input(quake_data)
        if selection == "2":
            prompt_property_input(quake_data)


def prompt_location_input(quake_data):
    latitude = input("Please supply a latitude: ")
    longitude = input("Please supply a longitude: ")
    distance = input("Please supply a distance to compare: ")

    quake_data.set_location_filter(latitude, longitude, distance)

def prompt_property_input(quake_data):
    pass

def read_json(filepath="../Data/earthquakes.geojson"):
    filepath = Path(filepath)
    geojson = {}
    if filepath.exists():
        try:
            geojson = json.loads(filepath.read_text())
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        raise FileExistsError("File supplied does not exist!")
    return geojson


start_menu()
