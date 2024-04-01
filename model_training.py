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



# Transforms images to tensors and normalizes them
# Preparing images for working with convolutional neural networks (CNNs). 

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize images to 224x224
    transforms.ToTensor(), # convert PIL image to tensor, PyTorch tensors are multi-dimensional arrays that can be efficiently processed on GPUs
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize images,it helps stabilize and speed up the training process.
])

  
dataset = DogBreedDataset(root_dir='/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset', labels_file='/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/labels.csv', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Create training and validation dataloaders

# Training Batch 
train_loader = DataLoader(train_dataset, batch_size=70, shuffle=True, num_workers=4)

# Validation Batch 
val_loader = DataLoader(val_dataset, batch_size=70, shuffle=False, num_workers=4)

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


# Loss function : use to measure how well the model is performing ( how close the model's predictions are to the actual labels)
# Optimizer : use to update the model parameters to minimize the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = "cpu" 
model.to(device)

run_train_loop = False # Set to True to run the training loop
if run_train_loop:

    # For Training Batch
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size
    total_samples = num_batches * batch_size

    # For Validation Batch
    num_batches = len(val_loader)
    batch_size = val_loader.batch_size
    total_samples = num_batches * batch_size
    
    # Train the model
    if __name__ == '__main__':
        num_epochs = 1  # Set the total number of desired epochs
        iteration_count = 0
        evaluation_count = 0
        save_interval = 1
        start_epoch = 0

        # Optionally, load the model from a checkpoint if available
        checkpoint_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/model_checkpoint_epoch_6.pth'  # Update with the correct checkpoint path
       
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            iteration_count = checkpoint['iteration_count']
            evaluation_count = checkpoint['evaluation_count']

            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            print("--------------Epoch---------------------", epoch)
            model.train()
            print("--------------Starting Training---------------------")

            for inputs, labels in train_loader:
                print("--------------Iteration---------------------", iteration_count)
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
                    print("--------------Validation---------------------", evaluation_count)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    evaluation_count += 1

            accuracy = total_correct / total_samples
            print(f'Epoch [{epoch}/{start_epoch + num_epochs - 1}], Validation Accuracy: {accuracy:.4f}')

            # Saving the model checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'iteration_count': iteration_count,
                    'evaluation_count': evaluation_count,
                }, f'model_checkpoint_epoch_{epoch + 1}.pth')

    # Save the model
    torch.save(model.state_dict(), 'dog_breed_model.pth')

    #last model accuracy (model checkpoint 6) =  0.9754
