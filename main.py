import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DATA_DIR = '/home/est_posgrado_manuel.suarez/data/cats-faces/'

print(os.listdir(DATA_DIR))

print("Done!")