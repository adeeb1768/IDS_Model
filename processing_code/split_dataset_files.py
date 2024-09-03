import os

# Define the folder path
folder_path = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023" 

# Define the group size
group_size = 8

# Get all files in the folder
files = os.listdir(folder_path)

# Initialize a counter and an empty list to store groups
current_group = 0
group_files = []

for file in files:
  # Add the file to the current group
  group_files.append(file)

  # Check if we reached the group size
  if len(group_files) == group_size:
    # Create a new directory for the group
    group_dir = os.path.join(folder_path, f"group_{current_group + 1}")
    os.makedirs(group_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Move the files in the group to the new directory
    for group_file in group_files:
      source = os.path.join(folder_path, group_file)
      destination = os.path.join(group_dir, group_file)
      os.replace(source, destination)

    # Clear the group_files list and increment the counter
    group_files = []
    current_group += 1

# Handle any remaining files (less than the group size)
if group_files:
  # Create a directory for remaining files (optional)
  remaining_dir = os.path.join(folder_path, "remaining_files")
  os.makedirs(remaining_dir, exist_ok=True)

  # Move remaining files to the directory
  for group_file in group_files:
    source = os.path.join(folder_path, group_file)
    destination = os.path.join(remaining_dir, group_file)
    os.replace(source, destination)

print(f"Successfully processed {len(files)} files. Groups created: {current_group + 1}")
