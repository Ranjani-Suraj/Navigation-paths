import googlemaps
import os
from dotenv import load_dotenv
from pathlib import Path
import json


def get_gmaps_client():
    path = Path("Env/.env")
    load_dotenv(dotenv_path=path)
    return googlemaps.Client(key='AIzaSyBeE0j7ZCad54Ec7Njq9PEwv4qmYqUF2qY')


def get_distance_matrix(client, origins, destinations):
    return client.distance_matrix(origins, destinations, mode="driving", departure_time="now")


def get_directions(client, origin, destination):
    return client.directions(origin, destination, mode="driving", departure_time="now")


def distance_matrix_parse_json_dictionary():
    individual_statements = {}
    addresses = []
    with open("distance_matrix_raw_output.json") as json_file:
        data = json.load(json_file)
        addresses.append(data['origin_addresses'])
        i = 0
        for origin in data['origin_addresses']:
            j = 0
            for destination in data['destination_addresses']:
                distance = data['rows'][i]['elements'][j]['distance']['text']
                duration = data['rows'][i]['elements'][j]['duration']['text']
                individual_statements[f"{origin}_{destination}"] = {
                    "origin": origin,
                    "destination": destination,
                    "params": {"distance": distance, "duration": duration}
                }
                j += 1
            i += 1
    return [individual_statements, addresses]


def directions_parse_json_dictionary():
    statements = {}
    individual_statements = []
    with open("directions_raw_output.json") as json_file:
        data = json.load(json_file)
        for i in range(len(data)):
            for j in range(len(data[i][0]['legs'][0]['steps'])):
                try:
                    individual_statements.append([data[i][0]['legs'][0]['steps'][j]['start_location'],
                                                  data[i][0]['legs'][0]['steps'][j]['distance']['text']])
                except:
                    print("Not found")
        return individual_statements


def export_json(parsed_json_data):
    with open("sample.json", "w") as outfile:
        json.dump(parsed_json_data, outfile)


def run(locations):
    client = get_gmaps_client()
    origins = locations
    destinations = locations

    distance_matrix = get_distance_matrix(client, origins, destinations)

    with open("distance_matrix_raw_output.json", "w") as outfile:
        json.dump(distance_matrix, outfile)
    outfile.close()
    directions = []
    for origin in origins:
        for destination in destinations:
            directions.append(get_directions(client, origin, destination))

    with open("directions_raw_output.json", "w") as outfile:
        json.dump(directions, outfile)
    outfile.close()
