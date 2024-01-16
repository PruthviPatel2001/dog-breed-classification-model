import pandas as pd

# Assuming your CSV file is named 'dogs.csv'
file_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/labels.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract distinct breeds
distinct_breeds = df['breed'].unique()
# Convert the result to a list if needed
distinct_breeds_list = sorted(list(distinct_breeds))

print(distinct_breeds_list[29])
