import gmaps_api_ops

# Run the initializer
gmaps_api_ops.run(["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA"])

# Parse the specific JSON data
parsed = gmaps_api_ops.directions_parse_json_dictionary()

# Parsed data is ready for use!
print(parsed)