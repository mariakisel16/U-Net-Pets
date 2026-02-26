import os
import random
import math
from collections import Counter
from typing import Callable, Optional, Union, Any
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset # Added this import

class PetDataset(Dataset):
  def __init__(self, root = "kaggle/input/the-oxforddriiit-pet-dataset", is_train = True, transform = None):
    self.transform = transform
    self.classes = ["background", "animal"]
    self.root = root
    self._images_dir = os.path.join(self.root, "images", "images") # Defined _images_dir
    self._annotations_dir = os.path.join(self.root, "annotations", "annotations") # Defined _annotations_dir

    #Select the appropriate annotation file based on whether it's training or test
    if is_train:
      annotations_file = os.path.join(self._annotations_dir, "trainval.txt") # Corrected variable name and path
    else:
      annotations_file = os.path.join(self._annotations_dir, "test.txt") # Corrected variable name and path

    #Read the annotation file and extract image names
    with open(annotations_file, 'r') as file: # Used corrected variable name
      # Assuming image name is the first word on each line, split by space
      self.img_names = [line.split(' ')[0] for line in file.readlines()]

  def __len__ (self): # Moved outside __init__ and corrected indentation
    return len(self.img_names)

  def __getitem__ (self, item): # Moved outside __init__ and corrected indentation
    img_name = self.img_names[item] # Corrected from self.image_names
    img_path = os.path.join(self._images_dir, img_name) + ".jpg" # Corrected os.path.joing and path
    mask_path = os.path.join(self._annotations_dir, "trimaps", img_name) + ".png" # Corrected path

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    # Map trimap values to target classes (0 for background, 1 for animal)
    # Original: 1=pet, 2=border, 3=background
    # Target: 0=background, 1=animal
    mask[mask == 2] = 0 # Map border to background
    mask[mask == 1] = 1 # Map pet to animal
    mask[mask == 3] = 0 # Map original background to background

    if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].long()

    return image, mask