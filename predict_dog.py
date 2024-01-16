import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
from utils.getclasses import distinct_breeds_list
import matplotlib.pyplot as plt


model = models.resnet50(pretrained=True)
num_classes = len(distinct_breeds_list)  
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the class labels
classes =  distinct_breeds_list

# Load the model from the checkpoint
checkpoint_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/model_checkpoint_epoch_2.pth'  # Update with the correct checkpoint path
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image_label(image_path, model, transform):
    # Load and preprocess the input image
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Use the model for prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    print("----------",predicted_idx,"----------")
    # Convert the class index to the label (assuming you have a list of classes/breeds)
    predicted_label = classes[predicted_idx.item()]  # Update 'classes' with your list of labels

    plt.imshow(input_image)
    plt.title(f'Predicted Label: {predicted_label}')
    plt.show()

# Example usage
image_path = '/Users/pruthvipatel/Documents/projects/dog_breed_classification/dataset/new_data/images/be575caa5b993cf44f40ac8193db9597.jpg'  # Replace with the path to your image
predicted_label = predict_image_label(image_path, model, transform)
