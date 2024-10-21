from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import pandas as pd


def get_cities_within_radius(latitude, longitude, radius, cities_df, max_rows=10):
    # Calculate the distance from the input coordinates to each city
    cities_df['distance'] = cities_df.apply(
        lambda row: geodesic((latitude, longitude), (row['lat'], row['lng'])).km,
        axis=1
    )

    # Filter cities within the specified radius
    nearby_cities = cities_df[cities_df['distance'] <= radius].head(max_rows)

    return nearby_cities['city'].tolist()


# Example usage
latitude = 38.840076  # Example latitude (San Francisco)
longitude = -77.303729  # Example longitude (San Francisco)
radius = 1130  # Example radius in kilometers

# Load the SimpleMaps World Cities Database (replace with your path)
cities_df = pd.read_csv('Data/worldcities.csv')

# Create a geocoder instance (using Nominatim from geopy)
geolocator = Nominatim(user_agent="city_locator")

# Geocode the input coordinates to get the city name
location = geolocator.reverse((latitude, longitude), language='en')
input_city = location.raw['address']['city']

# Filter the dataset to exclude the input city
cities_df = cities_df[cities_df['city'] != input_city]

# Call the function to get cities within the specified radius
cities_within_radius = get_cities_within_radius(latitude, longitude, radius, cities_df)

# Print the result
if cities_within_radius:
    print(f"Cities within {radius} km of {input_city}:")
    for city in cities_within_radius:
        print(city)
else:
    print("No cities found within the specified radius.")
