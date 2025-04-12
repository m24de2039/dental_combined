import torch
import torch.nn as nn
from torchvision import transforms


# Your trained model architecture
class DentalModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_output_size = hidden_units * (500 // 4) * (500 // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, output_shape)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        return self.classifier(x)


def load_model(model_path):
    model = DentalModel(input_shape=3, hidden_units=7, output_shape=7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_category(model, image):
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    class_names = [
        'Pulpitis',
        'Bony Impaction',
        'Improper Restoration with Chronic Apical Periodontitis',
        'Chronic Apical Periodontitis with Vertical Bone Loss',
        'Embedded Tooth',
        'Dental Caries',
        'Periodontitis'
    ]
    return class_names[predicted.item()]
