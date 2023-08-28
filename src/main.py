import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description='Classify an image using a pre-trained model.')
parser.add_argument('image_path', type=str, help='Path to the input image.')
args = parser.parse_args()
image_path = args.image_path

model = models.resnet18(weights=None)  # Set pretrained to False because we're going to load our own weights

# Modify the fully connected layer to match the number of classes for CIFAR-10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

# Now load the saved weights
model.load_state_dict(torch.load('cifar10_resnet18.pth'))

model.eval()  # Set the model to evaluation mode

# Image transformation (assuming you used these transformations during training)
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load and transform image
image = Image.open(image_path).convert("RGB")
image = transform(image)

# Add batch dimension and run the model
image = image.unsqueeze(0)  # Add batch dimension. image shape becomes [1, 3, 224, 224]

with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"The model predicts the image as: {classes[predicted_class]}")