import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import numpy as np
import cv2
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter


# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available.")

augs = [RandomHorizontalFlip(), RandomRotation(10), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
# transform function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.augment_transforms = transforms.Compose(augs)
        self.images = []
        self.labels = []
        self.classes = []  # Store unique class labels
        
        # Load images, labels, and classes from the folder_path
        self.load_data()
        
    def load_data(self):
        subfolders = ['Cyst', 'Normal', 'Stone', 'Tumor']
        
        for folder_name in subfolders:
            folder_path = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(folder_path):
                raise ValueError(f"{folder_path} does not exist.")
            
            images = os.listdir(folder_path)
            for image_name in images:
                image_path = os.path.join(folder_path, image_name)
                self.images.append(image_path)
                
                if folder_name not in self.classes:
                    self.classes.append(folder_name)
                
                self.labels.append(self.classes.index(folder_name))
    
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        
        # Load the image using PIL
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)  # Convert label to tensor
    
    def __len__(self):
        return len(self.images)


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Create the CNN model
model = CNN(num_classes=4)
model = model.to(device)

# Save the trained model
train_folder = "C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\train"
batch_size = 32
shuffle = True

train_dataset = CustomDataset(train_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
'''
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

weights_dir = "C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\weights"
os.makedirs(weights_dir, exist_ok=True)
model = model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
         
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Save the model's state after each epoch
    weights_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

#------------------------------------------------------ By using trained model weights-------------------------------------------#
#testing

# Load the test dataset
test_folder = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\test'
test_dataset = CustomDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define the directory path to the saved model weights
weights_dir = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\training_weights'
# Define the directory path to save the best model
save_dir = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\best_model'

# Initialize variables to store the best model and its performance metric
best_model = None
best_accuracy = 0.0

# Iterate over the model weight files in the directory
for file in os.listdir(weights_dir):
    if file.endswith('.pth'):
        model_path = os.path.join(weights_dir, file)
        
        # Instantiate the model
        model = CNN(num_classes=len(test_dataset.classes))
        model = model.to(device)
        
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Evaluate the model on the test data
        test_predictions = []
        test_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # Calculate the accuracy for the current model
        accuracy = accuracy_score(test_labels, test_predictions)
        print(f"Model: {file}, Accuracy: {accuracy}")
        
        # Update the best model if the current model performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        
# Print the accuracy of the best model
print(f"Best Model Accuracy: {best_accuracy}")

# Save the best model to the specified directory
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'best_model.pth')
torch.save(best_model.state_dict(), save_path)

'''
'''
############################################## confusion matrix ###################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

test_folder = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\test'
test_dataset = CustomDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define the directory path to the saved model weights
weights_dir = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\training_weights'
# Define the directory path to save the best model


# Initialize variables to store the best model and its performance metric
best_model = None
best_accuracy = 0.0

# Iterate over the model weight files in the directory
for file in os.listdir(weights_dir):
    if file.endswith('.pth'):
        model_path = os.path.join(weights_dir, file)
        
        # Instantiate the model
        model = CNN(num_classes=len(test_dataset.classes))
        model = model.to(device)
        
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        model.eval()


true_labels = []
predicted_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
cm = confusion_matrix(true_labels, predicted_labels)
class_names = test_dataset.classes
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
'''

#------------------------------ testing for unseen data using softmax-----------------------------------------------#

import torch.nn.functional as F

test_image_path = 'C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\Stone- (17).jpg'
test_image = Image.open(test_image_path).convert('L')

# Apply the transformations to the test image
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_image = test_transform(test_image)

# Add a batch dimension and move the test image to the same device as the model
test_image = test_image.unsqueeze(0).to(device)

# Set the model to evaluation mode
best_model_path='C:\\Users\\ACER LAPTOP\\Documents\\DATASET_\\Kindey_Stone_Dataset_with_split\\best_model\\best_model.pth'
model = CNN(num_classes=4)
model.load_state_dict(torch.load(best_model_path))
model = model.to(device)
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    # Forward pass through the model
    output = model(test_image)
    softmax_output = F.softmax(output, dim=1)
    _, predicted = torch.max(softmax_output, 1)

# Convert the predicted label to a human-readable class name
class_names = train_dataset.classes
predicted_class = class_names[predicted.item()]

# Print the predicted class
#print(f"Prediction: {predicted_class}")
#################################################################################

import matplotlib.pyplot as plt
plt.imshow(test_image.cpu().squeeze(), cmap='gray')
plt.title(f"Prediction: {predicted_class}")
plt.axis('off')
plt.show()
