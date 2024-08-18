# import h5py
# import tensorflow as tf
# from tensorflow.keras.layers import BatchNormalization
import json
import deepforest
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import os
from PIL import Image
from PIL import ImageDraw
import numpy as np

# Define the MLP head with two outputs
class MLPRegressionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super(MLPRegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Output two values: AGB and Carbon Content

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)  # Two outputs, one for AGB and one for Carbon Content
        return outputs

# Define the MLP for processing additional features
class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Define the modified model that combines the backbone and additional features
class ModifiedDeepForestAGB_CC(nn.Module):
    def __init__(self, deepforest_model, mlp_hidden_dim=256, feature_dim=12, feature_hidden_dim=128):
        super(ModifiedDeepForestAGB_CC, self).__init__()
        self.backbone = deepforest_model.model.backbone

        # Assuming the output from the FPN is 256 channels with some spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        backbone_features = 256  # Number of output channels from the FPN

        # MLP for processing additional features
        self.feature_mlp = FeatureMLP(input_dim=feature_dim, hidden_dim=feature_hidden_dim)

        # Total features after concatenation
        total_features = backbone_features + feature_hidden_dim

        # Define the MLP regression head with two outputs
        self.mlp_head = MLPRegressionHead(in_features=total_features, hidden_dim=mlp_hidden_dim)

    def forward(self, x, additional_features):
        # Extract features using the backbone
        backbone_features = self.backbone(x)['0']  # Access the first layer's output

        # Apply average pooling and flatten the feature map
        pooled_backbone_features = self.avgpool(backbone_features)
        flattened_backbone_features = torch.flatten(pooled_backbone_features, 1)

        # Process additional features through the MLP
        processed_features = self.feature_mlp(additional_features)

        # Concatenate the backbone features with the processed additional features
        combined_features = torch.cat([flattened_backbone_features, processed_features], dim=1)

        # Pass the combined features through the MLP regression head
        agb_cc_outputs = self.mlp_head(combined_features)

        # Split the outputs into AGB and Carbon Content
        agb_output, cc_output = agb_cc_outputs[:, 0], agb_cc_outputs[:, 1]

        return agb_output, cc_output
    
class TreeDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the row data
        row = self.data.iloc[idx]
        
        # Load the image
        img_path = os.path.join(self.img_root, row['img_name'], row['img_path'])
        image = Image.open(img_path).convert("RGB")
        
        # Crop the image using xmin, ymin, xmax, ymax
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        print(img_path)
        
        # Apply any transformations
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # Calculate area using Xmin, Ymin, Xmax, Ymax
        area = (row['Xmax'] - row['Xmin']) * (row['Ymax'] - row['Ymin'])
        
        # Extract features
        features = torch.tensor([
            row['xmin'],
            row['ymin'],
            row['xmax'],
            row['ymax'],
            row['score'],
            row['is_musacea_d'],
            row['is_banana'],
            row['diameter'],
            row['height'],
            row['updated diameter'],
            row['updated height'],
            area
        ], dtype=torch.float32)
        
        # Extract labels
        labels = torch.tensor([row['AGB'], row['carbon']], dtype=torch.float32)
        
        return cropped_image, features, labels, [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])], img_path


# def remove_freeze_attribute(model_path):
#     with h5py.File(model_path, 'r+') as f:
#         # Read the model configuration
#         model_config = f.attrs['model_config']
        
#         # Ensure the model_config is a string if it's not already
#         if isinstance(model_config, bytes):
#             model_config = model_config.decode('utf-8')

#         # Validate JSON before modification
#         try:
#             json.loads(model_config)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON in model_config before modification: {e}")


#         # Replace the freeze attribute
#         model_config = model_config.replace('"freeze": true,', '"freeze": True,')
#         model_config = model_config.replace(', "freeze": true', ', "freeze": True')

#         # Validate JSON after modification
#         try:
#             json.loads(model_config)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON in model_config after modification: {e}")
        
#         # Write the updated configuration back
#         f.attrs['model_config'] = model_config.encode('utf-8')
        

# def load_model_with_hdf5():
#     # Path to your model
#     model_path = 'model/deepforest/final_model_4000_epochs_35.h5'

#     # Remove the freeze attribute from the model configuration
#     remove_freeze_attribute(model_path)

#     # Now load the model
#     model = tf.keras.models.load_model(model_path)

#     # Summarize the model architecture
#     model.summary()

#     return model



# Function to load and modify the DeepForest model
def load_modified_deepforest_model():
    model = main.deepforest()
    model.use_release()

    # Define the feature dimension (e.g., 12 for your specified features)
    feature_dim = 12
    
    # Instantiate the modified model
    modified_model = ModifiedDeepForestAGB_CC(model, feature_dim=feature_dim)
    
    return modified_model


if __name__ == '__main__':

    model = load_modified_deepforest_model()

    csv_file = 'data/mapping/final_dataset.csv'
    img_root = 'data/tiles'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a dataset object
    dataset = TreeDataset(csv_file, img_root)

    