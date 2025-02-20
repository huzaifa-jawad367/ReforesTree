import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Enhanced MLP for processing additional features
class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout // 2)
        )

    def forward(self, x):
        return self.mlp(x)


# ResNet-specific regression head
class MLPRegressionHead_Resnet(nn.Module):
    def __init__(self, in_features=512, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Two outputs: AGB and Carbon Content
        )

    def forward(self, x):
        return self.head(x)


# Full model with ResNet-18 backbone
class ModifiedDeepForestAGB_CC_Resnet(nn.Module):
    def __init__(self, feature_dim=12, feature_hidden_dim=256, freeze_backbone=True):
        super().__init__()
        
        # ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head
        
        # Feature processing MLP
        self.feature_mlp = FeatureMLP(input_dim=feature_dim, hidden_dim=feature_hidden_dim)
        
        # Regression head
        self.regression_head = MLPRegressionHead_Resnet(in_features=512 + feature_hidden_dim)
        
        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last two blocks for fine-tuning
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

    def forward(self, x, additional_features):
        # Extract image features
        img_features = self.backbone(x)
        
        # Process additional features
        processed_features = self.feature_mlp(additional_features)
        
        # Concatenate features
        combined_features = torch.cat([img_features, processed_features], dim=1)
        
        # Predict AGB and Carbon Content
        outputs = self.regression_head(combined_features)
        agb_output, cc_output = outputs[:, 0], outputs[:, 1]
        
        return agb_output, cc_output
