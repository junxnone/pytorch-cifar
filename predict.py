import torch
import torchvision
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--test_data_path', '-ted', default='data/test', type=str, help='test data path')
parser.add_argument('--input_image', '-i', default='none', type=str, help='test image')
args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
model_file = "checkpoint/ckpt.pth"
model  = torch.load(model_file, map_location='cpu')
model = model.to(device)
model.eval()

def predict_dataset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.ImageFolder(args.test_data_path, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        print(f'predict: {classes[pred[0][0]]} - label: {classes[targets[0]]}')

def predict_image(file_path):
    image = torch.tensor(np.array(Image.open(file_path)))
    image = torch.reshape(image,(1,3,32,32)).float()
    input = image.to(device)
    output = model(input)
    _, pred = output.topk(5, 1, largest=True, sorted=True)
    print(f'predict: {classes[pred[0][0]]}')

if __name__ == '__main__':

    #predict_dataset(testloader)
    predict_image(args.input_image)