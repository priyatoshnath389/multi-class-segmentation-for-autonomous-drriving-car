import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.module):
