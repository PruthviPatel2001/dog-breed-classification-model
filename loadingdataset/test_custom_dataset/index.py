import time
import torch
import matplotlib.pyplot as plt


def test_dog_breed_dataset(dataset):
    # Get a single item from the dataset
    sample_image, sample_label = dataset[0]

    # Convert the tensor to a NumPy array and display the image
    sample_image_np = sample_image.permute(1, 2, 0).numpy()
    plt.imshow(sample_image_np)
    plt.title(f'Label: {sample_label}, Breed: {dataset.classes[sample_label]}')
    plt.show()

