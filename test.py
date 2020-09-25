from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image 
from models.resnet import *
classes = ['1', '10', '15', '16', '14','12','9','2','3','6','17','11','4','18','8','13','5','7']
def test(net,image_file):
    input = Image.open(image_file).convert('RGB')
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = data_transforms(input)
    image = image.unsqueeze(0)
    image = image.to(device)
    outputs = net(image)
    _, preds = torch.max(outputs, 1)

    print("class is {}".format(classes[preds]))

    '''
    plt.figure("Image")
    plt.imshow(input)
    plt.axis('on')
    plt.title('predicted: {}'.format(classes[preds[0]]))
    save_path = os.path.join('save',os.path.basename(image_file))
    plt.savefig(save_path)
    #plt.show()
    '''

if __name__ == "__main__":
    save_path = './oil_net.pt'
    net = DefaultResnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    class_size = len(classes)
    net.fc = nn.Linear(num_ftrs, class_size)

    od_state_dict = torch.load(save_path)
    new_state_dict = {k[7:]:v for k,v in od_state_dict.items()} #remove module
    net.load_state_dict(new_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    net.eval()
    image_folder = '/home/jl/datasets/oilrecognition/val/1'
    image_list = [image for image in os.listdir(image_folder) if os.path.splitext(image)[1] in ['.png','.jpg']]
    for image in image_list:
        image_path = os.path.join(image_folder,image)
        test(net,image_path)
