# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchsummary import summary
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
from loadingdataset.index import DogBreedDataset
from loadingdataset.test_custom_dataset.index import test_dog_breed_dataset

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# Transforms images to tensors and normalizes them

# preparing images for working with convolutional neural networks (CNNs). 

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize images to 224x224
    transforms.ToTensor(), # convert PIL image to tensor, PyTorch tensors are multi-dimensional arrays that can be efficiently processed on GPUs
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize images,it helps stabilize and speed up the training process.
])

  
dataset = DogBreedDataset(root_dir='/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset', labels_file='/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/labels.csv', transform=transform)

# To preview a sample image from the dataset.
# test_dog_breed_dataset(dataset)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Create training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=70, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=70, shuffle=False, num_workers=4)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Using pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Freeze all layers except the final classification layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final classification layer for the number of dog breeds
num_classes = len(dataset.classes)

# Access final fully-connected layer
# Replacing the last layer of pre-trained classes with the classes of dog breeds
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
# Loss function : use to measure how well the model is performing ( how close the model's predictions are to the actual labels)
# Optimizer : use to update the model parameters to minimize the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = "cpu" 
model.to(device)


# Print model summary
# summary(model, input_size=(3, 224, 224))

num_batches = len(train_loader)
batch_size = train_loader.batch_size
total_samples = num_batches * batch_size
print(f"Number of samples: {total_samples}")
print(f"Number of batches: {num_batches}")
print(f"Batch size: {batch_size}")

# for val loder
num_batches = len(val_loader)
batch_size = val_loader.batch_size
total_samples = num_batches * batch_size
print(f"Number of samples in val: {total_samples}")
print(f"Number of batches in val : {num_batches}")
print(f"Batch size of val: {batch_size}")

# Train the model
if __name__ == '__main__':
    num_epochs = 1  # Set the total number of desired epochs
    iteration_count = 0
    evaluation_count = 0
    save_interval = 1
    start_epoch = 0

    # # Optionally, load the model from a checkpoint if available
    # checkpoint_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/model_checkpoint_epoch_1.pth'  # Update with the correct checkpoint path
    # if checkpoint_path:
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     iteration_count = checkpoint['iteration_count']
    #     evaluation_count = checkpoint['evaluation_count']
    #     start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    #     print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("--------------EPOCH---------------------", epoch)
        model.train()
        print("--------------Starting TRAINing---------------------")

        for inputs, labels in train_loader:
            print("--------------FOR---------------------", iteration_count)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iteration_count += 1

        print("--------------Starting Validation---------------------")
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                print("--------------VAL---------------------", evaluation_count)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                evaluation_count += 1

        accuracy = total_correct / total_samples
        print(f'Epoch [{epoch}/{start_epoch + num_epochs - 1}], Validation Accuracy: {accuracy:.4f}')

        # Save the model periodically, for example, after each epoch
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iteration_count': iteration_count,
                'evaluation_count': evaluation_count,
            }, f'model_checkpoint_epoch_{epoch + 1}.pth')

# Save the trained model after training completes
torch.save(model.state_dict(), 'dog_breed_model.pth')
