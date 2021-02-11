
from DCNN_dataset_maker.init import *

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# transform function for input's image processing
transform_SSD = transforms.Compose([
    transforms.Resize((int(image_size_SSD),int(image_size_SSD))),      # Resize the image to image_size x image_size pixels size.
    transforms.ToTensor(),       # Convert the image to PyTorch Tensor data type.
    transforms.Normalize(        # Normalize the image by adjusting
    mean=[0.485, 0.456, 0.406],  #  its average and
    std=[0.229, 0.224, 0.225]    #  its standard deviation at the specified values.              
    )])


# transform function for input's image processing
transform = transforms.Compose([
    transforms.Resize((int(image_size))),      # Resize the image to image_size x image_size pixels size.
    transforms.ToTensor(),       # Convert the image to PyTorch Tensor data type.
    transforms.Normalize(        # Normalize the image by adjusting
    mean=[0.485, 0.456, 0.406],  #  its average and
    std=[0.229, 0.224, 0.225]    #  its standard deviation at the specified values.              
    )])

image_dataset = ImageFolder(path_in, transform=transform) # save the dataset
image_dataset_SSD = ImageFolder(path_in, transform=transform_SSD) # save the dataset

# imports networks with weights
models = {} # get model's names

# Load SSD model from the checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
try:
    checkpoint = torch.load(checkpoint)
except:
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
pprint('Loaded SSD model')
model = checkpoint['model'].to(device).eval()


#models['vgg16'] = torchvision.models.vgg16(pretrained=True)
print("Loading pretrained torchvision's model..")
models['alex'] = torchvision.models.alexnet(pretrained=True)
models['vgg'] = torchvision.models.vgg16(pretrained=True)
models['mob'] = torchvision.models.mobilenet_v2(pretrained=True)
models['res'] = torchvision.models.resnext101_32x8d(pretrained=True)
pprint("Loaded!")

for name in models.keys():
    models[name].to(device).eval()

print(datetag, 'Running benchmark on host', HOST, 'with',device)
