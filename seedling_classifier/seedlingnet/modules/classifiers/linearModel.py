import torch
import torch.nn as nn

import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from model import RegressionModel

def meanStdFromTensor(tensor):
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, unbiased=False, keepdim=True)
    return mean, std

def normalizeTensor(tensor, mean, std):
    return ((tensor - mean)/std).float()

def calculate_mask_area(pil_mask):
    mask_array = np.array(pil_mask)
    area = np.sum(mask_array != 0)
    return area

def tensor2image(tensor):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(tensor)

def image2tensor(image):
    image2tensor_transformation = transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])

    return image2tensor_transformation(image)

def crop_image(image, bbox, outSize):
    # Read the image
    
    # Extract the coordinates from the bounding box
    x1, y1, x2, y2 = bbox
    x = (x1+x2)//2
    y = (y1+y2)//2
    
    # Calculate the region to be cropped around the center
    crop_left = x - outSize #max(0, )
    crop_upper = y - outSize #max(0, )
    crop_right = x + outSize #min(image.width, )
    crop_lower = y + outSize #min(image.height, )
    
    # Crop the image using the calculated region
    cropped_image = image.crop((crop_left, crop_upper, crop_right, crop_lower))
   
    return cropped_image

class LinearModel:
    def __init__(self, weights_path):
        self.model = torch.load(weights_path)
        self.meanArea = 19537.82421875 
        self.stdArea = 5348.92578125
        self.meanLength = 75.7072982788086 
        self.stdLength = 16.3155517578125
        self.treshold = 50
    
    @torch.no_grad()
    def classify(self, horizontalMask, verticalMask):
        horizontalMask = Image.fromarray(horizontalMask)
        horizontal_bbox = horizontalMask.getbbox()
        horizontal_mask_cropped = crop_image(horizontalMask, horizontal_bbox, 130)
        horizontal_area = np.array([float(calculate_mask_area(horizontal_mask_cropped))]).reshape(1,1)

        verticalMask = Image.fromarray(verticalMask)
        vertical_bbox = verticalMask.getbbox()
        vertical_mask_cropped = crop_image(verticalMask, vertical_bbox, 170)
        vertical_area = np.array([float(calculate_mask_area(vertical_mask_cropped))]).reshape(1,1)

        areaTotal = vertical_area + horizontal_area

        areaTotalNormalized = normalizeTensor(torch.tensor(areaTotal), torch.tensor(self.meanArea), torch.tensor(self.stdArea))
        lengthNormalized = self.model(areaTotalNormalized)
        estimatedLength = lengthNormalized*self.stdLength + self.meanLength
                
        return estimatedLength.item() > self.treshold
    


    
