import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# 1. RECREATE THE STRUCTURE (Updated for modern torchvision)
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, 2)

# 2. LOAD THE SAVED WEIGHTS
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# 3. PREPARE DATA
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. RUN INFERENCE
img = Image.open("3.jpg")
input_tensor = transform(img).unsqueeze(0) 

# Define your labels based on how your folder was organized
# (e.g., if 'cats' was folder 0 and 'dogs' was folder 1)
class_names = ['Cat', 'Dog'] 

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()
    
    print(f"Prediction: {class_names[prediction]}")
    print(f"Confidence: {confidence:.2%}")