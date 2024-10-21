from geopy.distance import geodesic
import csv


def load_geonames_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        header = next(csvreader)  # Skip the header
        for row in csvreader:
            data.append({
                'geonameid': int(row[0]),
                'name': row[1],
                'latitude': float(row[4]),
                'longitude': float(row[5]),
            })
    return data


def find_cities_within_radius(data, center_latitude, center_longitude, radius):
    center_coordinates = (center_latitude, center_longitude)
    result = []

    for entry in data:
        city_coordinates = (entry['latitude'], entry['longitude'])
        distance = geodesic(center_coordinates, city_coordinates).kilometers

        if distance <= radius:
            result.append(entry)

    return result


# Example usage
file_path = 'C:/Users/samar/Downloads/US/US.txt'  # Replace with the actual path to your dataset
center_latitude = 40.454907  # Example latitude (San Francisco)
center_longitude = -79.958476  # Example longitude (San Francisco)
radius = 100  # Example radius in kilometers

geonames_data = load_geonames_data(file_path)
cities_within_radius = find_cities_within_radius(geonames_data, center_latitude, center_longitude, radius)

# Print the result
if cities_within_radius:
    print(f"Cities within {radius} km of the center coordinates:")
    for city in cities_within_radius:
        print(f"{city['name']} (Geoname ID: {city['geonameid']})")
else:
    print("No cities found within the specified radius.")
