from earthquakes import *
from pathlib import *
from nphelpers import printnp


def start_menu():
    """Main menu function used to display earthquake data menu"""
    print("Welcome to Earthquake Analyzer!")
    filename = input("Please supply a filename to analyze: ")
    geojson = read_json(f"../Data/{filename}")
    vectorized_data_to_quake_object = np.vectorize(data_to_quake_object())

    quake_data = QuakeData(geojson)
    while True:
        selection = input("Please select an option: \n (1) Set location filter \n (2) Set property filter"
                          " \n (3) Clear filters \n (4) Display Quakes \n (5) Display exceptional quakes "
                          "\n (6) Display magnitude stats \n (7) Plot quake map \n (8) Plot magnitude stats \n (9) Quit")
        if selection == "1":
            prompt_location_input(quake_data)
        elif selection == "2":
            prompt_property_input(quake_data)
        elif selection == "3":
            quake_data.clear_filters()
        elif selection == "4":
            quakes_objects = vectorized_data_to_quake_object(quake_data.get_filtered_array())
        elif selection == "5":
            pass
        elif selection == "6":
            pass
        elif selection == "7":
            pass
        elif selection == "8":
            pass
        elif selection == "9":
            break


def data_to_quake_object(filtered_array):
    quake_objects = []
    print(filtered_array)
    current_quake = filtered_array['quake']
    time = current_quake['properties']['time']
    q_type = current_quake['properties']['type']
    coords = (filtered_array['lat'], filtered_array['long'])
    quake_objects.append(Quake(filtered_array['magnitude'], time, filtered_array['felt'], filtered_array['significance'],q_type, coords))
    return quake_objects



def prompt_location_input(quake_data):
    latitude = input("Please supply a latitude: ")
    longitude = input("Please supply a longitude: ")
    distance = input("Please supply a distance to compare: ")

    quake_data.set_location_filter(latitude, longitude, distance)


def prompt_property_input(quake_data):

    print("Please supply a minimum of one argument for this operation")
    magnitude = input("Please supply the magnitude: ")
    felt = input("Please supply the felt: ")
    significance = input("Please supply the significance: ")

    if all(arg is None for arg in [magnitude, felt, significance]):
        print("WARNING: A single argument is required for this operation")

    quake_data.property_filter(magnitude, felt, significance)


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


# start_menu()
vectorized_data_to_quake_object = np.vectorize(data_to_quake_object)
geojson = read_json()
qd = QuakeData(geojson)
qd_objects = vectorized_data_to_quake_object(qd.get_filtered_array())
