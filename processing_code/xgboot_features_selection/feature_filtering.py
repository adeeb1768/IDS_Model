import pandas as pd
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Define the folder path and output foldere
folder_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\balanced_filtered"
output_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\ready_files"
# Function to manipulate a single CSV file using pandas
def manipulate_csv(filename):
  # Read the CSV file using pandas.read_csv
  data = pd.read_csv(os.path.join(folder_path, filename))
  
  # Process the data 
  filtered_data = data[['IAT', 'Rate', 'Header_Length', 'Number', 'flow_duration', 'syn_count', 'Variance','Protocol Type',
                         'Weight', 'Tot sum', 'rst_count', 'HTTPS', 'Min', 'Covariance', 'Max', 'Tot size', 'label']]

  # Write modified data to a new CSV file
  filtered_data.to_csv(os.path.join(output_path, f"filtered_{filename}"), index=False)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
  # Check if it's a CSV file
  if filename.endswith(".csv"):
    manipulate_csv(filename)
    print(f"Finished manipulating {filename}")

