import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

###############################################
# Define your network modules and dataset
###############################################

# Define the MLP head with two outputs
class MLPRegressionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super(MLPRegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Two outputs: AGB and Carbon Content

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)
        return outputs

# A variant for the ResNet model
class MLPRegressionHead_Resnet(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=256):
        super(MLPRegressionHead_Resnet, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# MLP to process additional features
class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Combined model: ResNet backbone + additional features
class ModifiedDeepForestAGB_CC_Resnet(nn.Module):
    def __init__(self, mlp_hidden_dim=256, feature_dim=12, feature_hidden_dim=128, freeze_backbone=True):
        super(ModifiedDeepForestAGB_CC_Resnet, self).__init__()
        # Use ResNet18 backbone (change to resnet50 if desired)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer

        # Number of backbone features (for resnet18, it's 512)
        backbone_features = 512

        # MLP for additional features
        self.feature_mlp = FeatureMLP(input_dim=feature_dim, hidden_dim=feature_hidden_dim)

        # Combined feature dimension
        total_features = backbone_features + feature_hidden_dim

        # Define regression head
        self.mlp_head = MLPRegressionHead_Resnet(in_features=total_features, hidden_dim=mlp_hidden_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x, additional_features):
        # Extract image features using the backbone
        backbone_out = self.backbone(x)
        # Process additional features
        processed_features = self.feature_mlp(additional_features)
        # Concatenate features
        combined_features = torch.cat([backbone_out, processed_features], dim=1)
        # Regression head outputs two values
        agb_cc_outputs = self.mlp_head(combined_features)
        agb_output, cc_output = agb_cc_outputs[:, 0], agb_cc_outputs[:, 1]
        return agb_output, cc_output

# Custom dataset class
class TreeDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Construct image path; adjust according to your CSV file columns
        img_path = os.path.join(self.img_root, row['img_name'], row['img_path'])
        image = Image.open(img_path).convert("RGB")
        # Crop image using given coordinates (ensure these columns exist in your CSV)
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            cropped_image = self.transform(cropped_image)
        # Calculate area using (Xmax - Xmin) * (Ymax - Ymin)
        area = (row['Xmax'] - row['Xmin']) * (row['Ymax'] - row['Ymin'])
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
        labels = torch.tensor([row['AGB'], row['carbon']], dtype=torch.float32)
        return cropped_image, features, labels

###############################################
# Define training, validation, and testing functions
###############################################

def train_model(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    running_mae_agb = 0.0
    running_mae_cc = 0.0

    for i, (images, features, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        agb_output, cc_output = model(images, features)
        loss_agb = criterion(agb_output, labels[:, 0])
        loss_cc = criterion(cc_output, labels[:, 1])
        loss = loss_agb + loss_cc

        loss.backward()
        optimizer.step()

        mae_agb = mean_absolute_error(labels[:, 0].cpu().detach().numpy(), agb_output.cpu().detach().numpy())
        mae_cc = mean_absolute_error(labels[:, 1].cpu().detach().numpy(), cc_output.cpu().detach().numpy())

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
        for i, (images, features, labels) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            agb_output, cc_output = model(images, features)
            loss_agb = criterion(agb_output, labels[:, 0])
            loss_cc = criterion(cc_output, labels[:, 1])
            loss = loss_agb + loss_cc

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

    all_agb_outputs = []
    all_cc_outputs = []
    all_agb_labels = []
    all_cc_labels = []

    with torch.no_grad():
        for images, features, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            agb_output, cc_output = model(images, features)
            loss_agb = criterion(agb_output, labels[:, 0])
            loss_cc = criterion(cc_output, labels[:, 1])
            loss = loss_agb + loss_cc

            running_loss += loss.item()
            all_agb_outputs.extend(agb_output.cpu().numpy())
            all_cc_outputs.extend(cc_output.cpu().numpy())
            all_agb_labels.extend(labels[:, 0].cpu().numpy())
            all_cc_labels.extend(labels[:, 1].cpu().numpy())

    mae_agb = mean_absolute_error(all_agb_labels, all_agb_outputs)
    mae_cc = mean_absolute_error(all_cc_labels, all_cc_outputs)
    epoch_loss = running_loss / len(test_loader)
    return epoch_loss, mae_agb, mae_cc

###############################################
# Define a function to run one fold training/testing
###############################################

def run_fold(fold, train_csv, val_csv, test_csv, img_root, model_save_dir, num_epochs=3):
    print(f"\n===== Running Fold {fold} =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Update img_root to the correct path for Kaggle
    #---------------update-
    
    train_dataset = TreeDataset(csv_file=train_csv, img_root=img_root, transform=transform)
    val_dataset = TreeDataset(csv_file=val_csv, img_root=img_root, transform=transform)
    test_dataset = TreeDataset(csv_file=test_csv, img_root=img_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize the model, loss, and optimizer
    model = ModifiedDeepForestAGB_CC_Resnet(freeze_backbone=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # TensorBoard writer for this fold
    writer = SummaryWriter(f"{model_save_dir}/runs/deepforest_agb_cc_fold_{fold}")
    
    best_val_loss = float('inf')
    
    # Training loop (3 epochs per fold)
    for epoch in range(num_epochs):
        print(f"\nFold {fold} - Epoch {epoch+1}/{num_epochs}")
        train_loss, train_mae_agb, train_mae_cc = train_model(model, train_loader, criterion, optimizer, device, writer, epoch)
        print(f"Train Loss: {train_loss:.4f} | MAE AGB: {train_mae_agb:.4f} | MAE CC: {train_mae_cc:.4f}")
        
        val_loss, val_mae_agb, val_mae_cc = validate_model(model, val_loader, criterion, device, writer, epoch)
        print(f"Val Loss: {val_loss:.4f} | MAE AGB: {val_mae_agb:.4f} | MAE CC: {val_mae_cc:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f"./{model_save_dir}/deepforest_agb_cc_best_fold_{fold}.pth"
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved:", model_save_path)
    
    writer.close()

    # Test the model
    test_loss, test_mae_agb, test_mae_cc = test_model(model, test_loader, criterion, device)
    print(f"\nTest Results for Fold {fold}: Loss: {test_loss:.4f}, MAE AGB: {test_mae_agb:.4f}, MAE CC: {test_mae_cc:.4f}")

###############################################
# Main: Loop over folds
###############################################

if __name__ == '__main__':
    # Define a dictionary for the folds with absolute CSV paths
    folds = {
        1: {
            "train": "./ReforesTree/data/split1/banana_splits_train_1.csv",
            "val":   "./ReforesTree/data/split1/banana_splits_val_1.csv",
            "test":  "./ReforesTree/data/split1/banana_splits_val_1.csv"
        },
        2: {
            "train": "./ReforesTree/data/split2/banana_splits_train_2.csv",
            "val":   "./ReforesTree/data/split2/banana_splits_val_2.csv",
            "test":  "./ReforesTree/data/split2/banana_splits_test_2.csv"
        },
        3: {
            "train": "./ReforesTree/data/split3/banana_splits_train_3.csv",
            "val":   "./ReforesTree/data/split3/banana_splits_val_3.csv",
            "test":  "./ReforesTree/data/split3/banana_splits_test_3.csv"
        },
        4: {
            "train": "./ReforesTree/data/split4/banana_splits_train_4.csv",
            "val":   "./ReforesTree/data/split4/banana_splits_val_4.csv",
            "test":  "./ReforesTree/data/split4/banana_splits_test_4.csv"
        },
        5: {
            "train": "./ReforesTree/data/split5/banana_splits_train_5.csv",
            "val":   "./ReforesTree/data/split5/banana_splits_val_5.csv",
            "test":  "./ReforesTree/data/split5/banana_splits_test_5.csv"
        }
    }

    # img_root = '/kaggle/input/reforest-dataset/tiles'
    img_root = 'dataset/tiles'

    model_save_dir = "Model_saves/Resnet18"

    # Loop through each fold and run training/testing
    for fold_num, paths in folds.items():
        run_fold(
            fold=fold_num,
            train_csv=paths["train"],
            val_csv=paths["val"],
            test_csv=paths["test"],
            img_root=img_root,
            model_save_dir=model_save_dir,
            num_epochs=30  # Run for 3 epochs per fold
        )
