import pandas as pd
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Define the folder path and output foldere
folder_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\combined"
output_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\balanced_filtered"
# Function to manipulate a single CSV file using pandas
def manipulate_csv(filename):
  # Read the CSV file using pandas.read_csv
  data = pd.read_csv(os.path.join(folder_path, filename))
  
  # Process the data 
  data=data[data['label'].isin(['DDoS-RSTFINFlood','DDoS-TCP_Flood','DDoS-ICMP_Flood','DoS-UDP_Flood','DoS-SYN_Flood','Mirai-greeth_flood',
                          'DDoS-SynonymousIP_Flood','Mirai-udpplain','DDoS-SYN_Flood','DDoS-PSHACK_Flood',
                          'DDoS-UDP_Flood','BenignTraffic'])]
  
  data = data.groupby('label', group_keys=False).apply(lambda x:x.sample(10000))

  # Write modified data to a new CSV file
  data.to_csv(os.path.join(output_path, f"modified_{filename}"), index=False)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
  # Check if it's a CSV file
  if filename.endswith(".csv"):
    manipulate_csv(filename)
    print(f"Finished manipulating {filename}")




