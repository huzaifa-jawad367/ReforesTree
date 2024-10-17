# import h5py
# import tensorflow as tf
# from tensorflow.keras.layers import BatchNormalization

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
import torchvision.models as models

import pandas as pd
import os
from PIL import Image
from PIL import ImageDraw
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

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
    
class MLPRegressionHead_Resnet(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=256):
        super(MLPRegressionHead_Resnet, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

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
    
# Define the modified model that combines the ResNet50 backbone and additional features
class ModifiedDeepForestAGB_CC_Resnet(nn.Module):
    def __init__(self, mlp_hidden_dim=256, feature_dim=12, feature_hidden_dim=128, freeze_backbone=True):
        super(ModifiedDeepForestAGB_CC_Resnet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # Remove the last layers of ResNet
        del self.backbone.fc

        # Assuming the output from the backbone is 2048 channels with some spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        backbone_features = 2048  # Number of output channels from ResNet50's last convolutional layer

        # MLP for processing additional features
        self.feature_mlp = FeatureMLP(input_dim=feature_dim, hidden_dim=feature_hidden_dim)

        # Total features after concatenation
        total_features = backbone_features + feature_hidden_dim

        # Define the MLP regression head with two outputs
        self.mlp_head = MLPRegressionHead_Resnet(in_features=total_features, hidden_dim=mlp_hidden_dim)

    def forward(self, x, additional_features):
        # Extract features using the backbone
        backbone_features = self.backbone(x)

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
        
        return cropped_image, features, labels
    
def train_model(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    running_mae_agb = 0.0
    running_mae_cc = 0.0

    for i, (images, features, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        agb_output, cc_output = model(images, features)
        loss_agb = criterion(agb_output, labels[:, 0])
        loss_cc = criterion(cc_output, labels[:, 1])
        loss = loss_agb + loss_cc

        # Calculate MAE
        mae_agb = mean_absolute_error(labels[:, 0].cpu().detach().numpy(), agb_output.cpu().detach().numpy())
        mae_cc = mean_absolute_error(labels[:, 1].cpu().detach().numpy(), cc_output.cpu().detach().numpy())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mae_agb += mae_agb
        running_mae_cc += mae_cc

        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)

    epoch_loss = running_loss / len(train_loader)
    epoch_mae_agb = running_mae_agb / len(train_loader)
    epoch_mae_cc = running_mae_cc / len(train_loader)
    return epoch_loss, epoch_mae_agb, epoch_mae_cc

def validate_model(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    running_mae_agb = 0.0
    running_mae_cc = 0.0
    
    with torch.no_grad():
        for i, (images, features, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            agb_output, cc_output = model(images, features)
            loss_agb = criterion(agb_output, labels[:, 0])
            loss_cc = criterion(cc_output, labels[:, 1])
            loss = loss_agb + loss_cc

            # Calculate MAE
            mae_agb = mean_absolute_error(labels[:, 0].cpu().detach().numpy(), agb_output.cpu().detach().numpy())
            mae_cc = mean_absolute_error(labels[:, 1].cpu().detach().numpy(), cc_output.cpu().detach().numpy())

            running_loss += loss.item()
            running_mae_agb += mae_agb
            running_mae_cc += mae_cc

            writer.add_scalar('Validation/Loss', loss.item(), epoch * len(val_loader) + i)
            writer.add_scalar('Validation/MAE_AGB', mae_agb, epoch * len(val_loader) + i)
            writer.add_scalar('Validation/MAE_CC', mae_cc, epoch * len(val_loader) + i)

    epoch_loss = running_loss / len(val_loader)
    epoch_mae_agb = running_mae_agb / len(val_loader)
    epoch_mae_cc = running_mae_cc / len(val_loader)

    return epoch_loss, epoch_mae_agb, epoch_mae_cc

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae_agb = 0.0
    running_mae_cc = 0.0

    all_agb_outputs = []
    all_cc_outputs = []
    all_agb_labels = []
    all_cc_labels = []

    with torch.no_grad():
        for images, features, labels in tqdm(test_loader):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            agb_output, cc_output = model(images, features)
            loss_agb = criterion(agb_output, labels[:, 0])
            loss_cc = criterion(cc_output, labels[:, 1])
            loss = loss_agb + loss_cc

            running_loss += loss.item()

            # Store outputs and labels for MAE calculation
            all_agb_outputs.extend(agb_output.cpu().numpy())
            all_cc_outputs.extend(cc_output.cpu().numpy())
            all_agb_labels.extend(labels[:, 0].cpu().numpy())
            all_cc_labels.extend(labels[:, 1].cpu().numpy())

    # Calculate MAE
    mae_agb = mean_absolute_error(all_agb_labels, all_agb_outputs)
    mae_cc = mean_absolute_error(all_cc_labels, all_cc_outputs)

    # Compute the overall loss
    epoch_loss = running_loss / len(test_loader)

    return epoch_loss, mae_agb, mae_cc

def run_testing():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = TreeDataset(csv_file='data/splits/test_dataset.csv', img_root='data/tiles', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load model
    model = load_modified_deepforest_model(freeze_backbone=True)
    model.load_state_dict(torch.load('Model_saves/deepforest_agb_cc_best.pth', map_location=device))  # Load the best model
    model = model.to(device)

    # Define loss function
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks

    # Test the model
    test_loss, test_mae_agb, test_mae_cc = test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test MAE AGB: {test_mae_agb:.4f}, Test MAE CC: {test_mae_cc:.4f}')




# Function to load and modify the DeepForest model
def load_modified_deepforest_model(freeze_backbone=True, model_name='deepforest'):
    if model_name == 'deepforest':
        model = main.deepforest()
        model.use_release()

    else:
        model = models.resnet50(pretrained=True)
        # Remove the last layer
        model = nn.Sequential(*list(model.children())[:-1])

    # Define the feature dimension (e.g., 12 for your specified features)
    feature_dim = 12
    
    # Instantiate the modified model
    modified_model = ModifiedDeepForestAGB_CC(model, feature_dim=feature_dim)

    # Freeze the backbone if specified
    if freeze_backbone:
        for param in modified_model.backbone.parameters():
            param.requires_grad = False
    
    return modified_model

def run_training():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = TreeDataset(csv_file='data/splits/train_dataset.csv', img_root='data/tiles', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    val_dataset = TreeDataset(csv_file='data/splits/val_dataset.csv', img_root='data/tiles', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # test_dataset = TreeDataset(csv_file='data/splits/test_dataset.csv', img_root='data/tiles', transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load model
    model = load_modified_deepforest_model(freeze_backbone=True)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tensorboard writer
    writer = SummaryWriter('runs_grad_bb/deepforest_agb_cc')

    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')    # Initialize best validation loss for model saving

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss, train_mae_agb, train_mae_cc = train_model(model, train_loader, criterion, optimizer, device, writer, epoch)
        print(f'Training Loss: {train_loss:.4f}, MAE AGB: {train_mae_agb:.4f}, MAE CC: {train_mae_cc:.4f}')
        
        # Validate the model
        val_loss, val_mae_agb, val_mae_cc = validate_model(model, val_loader, criterion, device, writer, epoch)
        print(f'Validation Loss: {val_loss:.4f}, MAE AGB: {val_mae_agb:.4f}, MAE CC: {val_mae_cc:.4f}')

        # Log epoch losses to TensorBoard
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Validation_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Train_MAE_AGB', train_mae_agb, epoch)
        writer.add_scalar('Epoch/Validation_MAE_AGB', val_mae_agb, epoch)
        writer.add_scalar('Epoch/Train_MAE_CC', train_mae_cc, epoch)
        writer.add_scalar('Epoch/Validation_MAE_CC', val_mae_cc, epoch)

        # Save the model if the validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'deepforest_agb_cc_best_2.pth')
            print('Best model saved!')

    writer.close()

if __name__ == '__main__':
    run_training()
    # model = models.resnet50(pretrained=True)
    # # remove model.fc
    # del model.fc
      
    # print(model)
    # run_testing()
