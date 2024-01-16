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
            print(f"Deleted {image_filename}")
        else:
            print(f"image not found")
            


def delete_images_except_selected_breeds(csv_file, image_folder, selected_breeds):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the image filename using the 'id' column
        image_filename = os.path.join(image_folder, f"{row['id']}.jpg")

        # Check if the breed is in the selected breeds list
        if row['breed'] in selected_breeds:
            print(f"Keeping image for {row['breed']}: {image_filename}")
        else:
            # Delete the image file if the breed is not in the selected breeds list
            print(f"Deleting image for {row['breed']}: {image_filename}")
            os.remove(image_filename)


# Provide the path to your CSV file and image folder
csv_file_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/new_data/labels.csv'
image_folder_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/new_data/images'
selected_breeds = ["siberian_husky", "pomeranian"]

# delete_images(csv_file_path, image_folder_path)
delete_images_except_selected_breeds(csv_file_path, image_folder_path, selected_breeds)