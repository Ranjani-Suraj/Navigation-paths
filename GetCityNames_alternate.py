import requests


def get_cities_within_radius(latitude, longitude, radius, max_rows=10):
    base_url = "http://api.geonames.org/findNearbyPlaceNameJSON"
    username = "samk1239"  # Replace with your GeoNames username

    params = {
        "lat": latitude,
        "lng": longitude,
        "radius": radius,
        "maxRows": max_rows,
        "username": username
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        cities = [place['name'] for place in data.get('geonames', [])]
        return cities
    else:
        print(f"Error: {response.status_code}")
        return None


# Example usage
latitude = 38.840076  # Example latitude (San Francisco)
longitude = -77.303729  # Example longitude (San Francisco)
radius = 400  # Example radius in kilometers

cities_within_radius = get_cities_within_radius(latitude, longitude, radius)

if cities_within_radius:
    print(f"Cities within {radius} km of the location:")
    for city in cities_within_radius:
        print(city)
else:
    print("Failed to retrieve city data.")
