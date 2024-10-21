import csv

# Read the input file and create a CSV file
with open('C:/Users/samar/Downloads/US/US.txt', 'r') as input_file, open('C:/Users/samar/Downloads/US/output_data.csv',
                                                                         'w', newline='') as output_file:
    # Create a CSV writer
    csv_writer = csv.writer(output_file, delimiter=',')

    # Write the header to the CSV file (if needed)
    csv_writer.writerow(
        ['nameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature class', 'feature code',
         'country code', 'cc2', 'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
         'dem', 'timezone', 'modification date'])

    # Process each line in the input file
    for line in input_file:
        # Split the line into fields based on tabs
        fields = line.strip().split('\t')

        # Write the fields to the CSV file
        csv_writer.writerow(fields)

print("Conversion complete. Output file: output_data.csv")
