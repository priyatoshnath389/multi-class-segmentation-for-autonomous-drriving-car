
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#Feature extractor (CNN-based)

class FeatureExtractor(nn.Module):
     def __init__(self, backbone="resnet18"): 
        super().__init__() 
        base_model = getattr(models, backbone)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

     def forward(self, x):
         return self.feature_extractor(x)

#CNN-based fusion module

class CNNFusion(nn.Module):
     def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.bn2 = nn.BatchNorm2d(out_channels)

     def forward(self, bev, rv):
         fused = torch.cat([bev, rv], dim=1)  # Concatenating along channel axis
         fused = F.relu(self.bn1(self.conv1(fused)))
         return F.relu(self.bn2(self.conv2(fused)))

#Detection & segmentation head

class DetectionHead(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#MVLidarNet Model

class MVLidarNet(nn.Module): 
    def __init__(self, num_classes=10): 
        super().__init__() 
        self.bev_extractor = FeatureExtractor()
        self.rv_extractor = FeatureExtractor() 
        self.fusion = CNNFusion(in_channels=1024, out_channels=512) 
        self.det_head = DetectionHead(512, num_classes)

    def forward(self, bev, rv):
        bev_feat = self.bev_extractor(bev)
        rv_feat = self.rv_extractor(rv)
        fused_feat = self.fusion(bev_feat, rv_feat)
        return self.det_head(fused_feat)

#Example usage

if __name__ == "main": 
    model = MVLidarNet(num_classes=10) 
    bev_input = torch.randn(1, 3, 224, 224)  # Simulated Bird's Eye View input rv_input = torch.randn(1, 3, 224, 224)   #
    output = model(bev_input, rv_input)
    print("Output shape:", output.shape)  # Should match (batch_size, num_classes, H, W)
