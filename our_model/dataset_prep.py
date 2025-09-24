import os
import shutil
import csv

# 🔹 Base directory where the four folders are located
base_dir = r'C:\Users\ASUS\Downloads\Raw Data\Raw'  # <<-- change this to your dataset location

# Define source folders and their labels
source_folders = {
    "meningioma": 1,
    "glioma": 2,
    "pituitary": 3,
    "no_tumor": 4
}

# Create destination folder inside base_dir
dest_folder = os.path.join(base_dir, "data")
os.makedirs(dest_folder, exist_ok=True)

# Path for CSV file
csv_file = os.path.join(base_dir, "Image_labels.csv")

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_path", "label"])  # header

    for folder, label in source_folders.items():
        folder_path = os.path.join(base_dir, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder {folder_path} not found, skipping...")
            continue

        # Get all images in this folder
        images = os.listdir(folder_path)

        for img in images:
            src_path = os.path.join(folder_path, img)
            dest_path = os.path.join(dest_folder, img)

            # Move the file
            shutil.move(src_path, dest_path)

            # Write row in CSV (store relative path from base_dir)
            writer.writerow([img, label])

print(f"✅ Done! Moved all images into {dest_folder} and created {csv_file}")
