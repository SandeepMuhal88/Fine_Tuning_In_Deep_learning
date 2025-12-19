import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models



vgg16=models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False

num_clases=15
vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)



def load_model():
    model=vgg16.to(device)
    model=model.load_state_dict(torch.load("fine_tuned_model.pth"))
    return model