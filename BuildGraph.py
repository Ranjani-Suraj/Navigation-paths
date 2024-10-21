import asyncio
import random

import googlemaps

import ConnectAPI
import gmaps_api_ops
# Main Utility Script

# Importing Scripts
import gmaps_api_ops
import BuildingGraph

# Google Maps Side of things
#gmaps_api_ops.run(["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA"])

# Building Graph, after which, you can do anything you want!

def making_weight_matrices(inputs):
    gmaps = gmaps_api_ops.get_gmaps_client()

    #input_string = input("Enter the places to be visited as a list of cities seperated by a space:/n")
    #places = input_string.split()
    #get landmarks on paths and run weather checking on them
    #check safety information of landmarks

    risk = list(list())
    cost = list(list())
    medium = list(list())
    gas_prices = {'AL':3.8, 'AK':3.6, 'AZ': 3.9, 'CA': 5.3, 'CO': 3.4, 'CT': 4.2, 'DC': 4.3,
                  'DE': 3.9, 'FL': 3.9, 'GA': 3.9, 'HI': 5.6, 'IA': 3.6, 'ID': 3.7,
                  'IL': 3.8, 'IN': 3.8, 'KA': 3.5, 'MA': 4.2, 'NY': 4.4, 'PA': 4.5,
                  'WI': 3.6, 'NJ': 4.0, 'RI': 4.2, 'VI': 3.9, 'WA': 4.7}


    # Run the initializer

    gmaps_api_ops.run(inputs)
    graph = BuildingGraph.build_graph()
    # print("graph:", graph)
    # Parse the specific JSON data
    parsed = gmaps_api_ops.directions_parse_json_dictionary()
    coordinates = list()
    # Parsed data is ready for use!
    # print("parsed", parsed)
    count = 0
    risk_set = []
    cost_set = []
    medium_set = []
    for i in range(0, len(inputs)):
        for j in range(0, len(inputs)):
            #find where the latitude longitudes start, then
            #(distance, time)
            #print("start")
            if(i!=j):
                LATITUDE = parsed[count][0]['lat']
                LONGITUDE  = parsed[count][0]['lng']
                coordinates.append({'lat': LATITUDE, 'lng': LONGITUDE})
                risks = asyncio.run(ConnectAPI.get_weather_details(LATITUDE, LONGITUDE))
                if risks is not None:
                    risk_set.append(risks[0] / 2.0 + risks[1] / 30.0)
                else:
                    risk_set.append(random.random()+random.random())

                address = gmaps.reverse_geocode({'lat': LATITUDE, 'lng': LONGITUDE})
                #print("address ", address)
                #state = address[0]['address_components'][5]['short_name']
                state = 'PA'
                # print("state ", state)

                #cost.append()
                if(len(state) == 2 and (state != 'US')):
                    cost_set.append(float(graph[i][j][0].split()[0].replace(',', '')) * gas_prices[state])
                else:
                    cost_set.append(float(graph[i][j][0].split()[0].replace(',', '')) * gas_prices['PA'])
                medium_set.append(risk_set[j]+cost_set[j]*1.5)

            else:
                risk_set.append(0)
                cost_set.append(0)
                medium_set.append(0)
                #print(risk)
            count += 1
        risk.append(risk_set)
        risk_set = []
        medium.append(medium_set)
        medium_set = []
        cost.append(cost_set)
        cost_set = []
        # print("cost:", cost)
        # print("risk", risk)
        # print("medium", medium)
    return [risk, cost, medium, coordinates]

making_weight_matrices(["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA", "New Jersey NJ", "New York NY", "Chicago IL", "Boston MA", "Allentown PA"])
              #"Gettysburg PA", "Bethlehem PA"])