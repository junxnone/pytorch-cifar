import torch
import torchvision
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
model_file = 'ckpt.pth'
model  = torch.load(model_file, map_location='cpu')
model = model.to(device)
model.eval()

def predict_image(file_path):
    image = torch.tensor(np.array(Image.open(file_path)))
    image = torch.reshape(image,(1,3,32,32)).float()
    input = image.to(device)
    output = model(input)
    _, pred = output.topk(5, 1, largest=True, sorted=True)
    result = f'predict: {classes[pred[0][0]]}'
    return result
