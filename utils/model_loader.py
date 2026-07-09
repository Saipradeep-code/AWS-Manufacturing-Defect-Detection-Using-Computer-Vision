import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path):
    # Load pretrained ResNet18
    model = models.resnet18(weights=None)

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 4)

    # Load trained weights
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )

    # Evaluation mode
    model.eval()

    return model