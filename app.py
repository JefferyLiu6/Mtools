import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the SmallNet model
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(3, 5, 3)  # Convolution layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.fc = nn.Linear(5 * 31 * 31, 8)  # Fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 5 * 31 * 31)  # Flatten the feature map
        x = self.fc(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x

# Load the pre-trained model
model_path = "./model/pytorch_model.bin"
class_labels = ["Gasoline_Can", "Hammer", "Pebbles", "Pliers", "Rope", "Screw_Driver", "Toolbox", "Wrench"]

# Initialize the model
net = SmallNet()

# Load state dictionary
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
net.load_state_dict(state_dict)
net.eval()  # Set the model to evaluation mode

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(128),  # Resize the shorter side to 128 pixels
    transforms.CenterCrop(128),  # Crop the center to get a 128x128 image
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Mapping class labels to three categories
class_map = {
    "Hammer": "hammer",
    "Rope": "rope",
    "Gasoline_Can": "other",
    "Pebbles": "other",
    "Pliers": "other",
    "Screw_Driver": "other",
    "Toolbox": "other",
    "Wrench": "other"
}

# Define the prediction function
def predict(image):
    # Apply the updated transformation pipeline
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = net(img_tensor)
        _, predicted = torch.max(outputs, 1)
        original_label = class_labels[predicted.item()]  # Get the original class label
        final_label = class_map[original_label]  # Map to hammer, rope, or other

    return final_label

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Gradio's image input
    outputs=gr.Textbox(),         # Gradio's textbox output
    title="Tool Classifier",
    description="Upload an image of a tool to classify it as hammer, rope, or other."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
