import os
import glob

# Define the main folder path
main_folder_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023" 

# Loop through each subfolder
for subfolder in os.listdir(main_folder_path):
  # Check if it's a subfolder (not a file)
  if os.path.isdir(os.path.join(main_folder_path, subfolder)):
    # Get all CSV files in the subfolder
    csv_files = glob.glob(os.path.join(main_folder_path, subfolder, "*.csv"))

    # Define the output filename within the subfolder (optional)
    output_filename = os.path.join(main_folder_path, subfolder, "combined.csv")  # Or a custom name

    # Initialize an empty list to store data
    combined_data = []

    # Loop through each CSV file in the subfolder
    for csv_file in csv_files:
      # Open the CSV file
      with open(csv_file, 'r') as f:
        # Read the CSV data
        data = f.readlines()
        # Skip the header row if it exists (optional)
        # data = data[1:] 
        # Combine the data from this file
        combined_data.extend(data)

    # Write the combined data to the output file
    with open(output_filename, 'w') as f:
      # Write the data to the file
      f.writelines(combined_data)

    print(f"Successfully concatenated CSV files in subfolder {subfolder} to {output_filename}")







