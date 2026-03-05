import os

# Path to dataset
base_path = "Data/raw/drawings"

# Classes
categories = ["spiral", "wave"]
splits = ["training", "testing"]
labels = ["healthy", "parkinson"]

for category in categories:
    for split in splits:
        for label in labels:
            folder_path = os.path.join(base_path, category, split, label)
            
            if os.path.exists(folder_path):
                count = len(os.listdir(folder_path))
                print(f"{category} | {split} | {label} : {count} images")
            else:
                print(f"Missing folder: {folder_path}")