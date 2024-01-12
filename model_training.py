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

# Create training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model summary
# summary(model, input_size=(3, 224, 224))

# Train the model
# num_epochs = 3
# # for epoch in range(num_epochs):
#     print("--------------EPOCH---------------------", epoch)
#     model.train()
#     print("--------------TRAIN---------------------")
#     for inputs, labels in train_loader:
#         print("--------------FOR---------------------")
#         inputs, labels = inputs.to(device), labels.to(device)

#         # Clears the gradients of all optimized parameters.
#         optimizer.zero_grad()
#         # Performs a forward pass to get the model predictions.
#         outputs = model(inputs)
#         # Calculates the loss based on the actual and predicted values.
#         loss = criterion(outputs, labels)
#         # Performs a backward pass to calculate the gradients.
#         loss.backward()
#         # Updates the parameters.
#         optimizer.step()

#     # Validation
#     model.eval()
#     total_correct = 0
#     total_samples = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total_samples += labels.size(0)
#             total_correct += (predicted == labels).sum().item()

#     accuracy = total_correct / total_samples
#     print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()