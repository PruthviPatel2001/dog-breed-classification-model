import os
import pandas as pd

def delete_images(csv_file, image_folder):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the image filename using the 'id' column
        image_filename = os.path.join(image_folder, f"{row['id']}.jpg")

        # Check if the image file exists, then delete it
        if os.path.exists(image_filename):
            os.remove(image_filename)
            print(f"Deleted image: {image_filename}")
        else:
            print(f"Image not found: {image_filename}")

# Provide the path to your CSV file and image folder
csv_file_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dogstoremove.csv'
image_folder_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/train'

delete_images(csv_file_path, image_folder_path)