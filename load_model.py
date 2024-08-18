import h5py
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import json
import deepforest
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Define the modified model that uses the new MLP head
class ModifiedDeepForestAGB_CC(nn.Module):
    def __init__(self, deepforest_model, mlp_hidden_dim=256):
        super(ModifiedDeepForestAGB_CC, self).__init__()
        self.backbone = deepforest_model.model.backbone

        # Assuming the output from the FPN is 256 channels with some spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 256  # Number of output channels from the FPN

        # Define the MLP regression head with two outputs
        self.mlp_head = MLPRegressionHead(in_features, mlp_hidden_dim)

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)['0']  # Access the first layer's output

        # Apply average pooling and flatten the feature map
        pooled_features = self.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)

        # Pass through the MLP regression head
        agb_cc_outputs = self.mlp_head(flattened_features)

        # Split the outputs into AGB and Carbon Content
        agb_output, cc_output = agb_cc_outputs[:, 0], agb_cc_outputs[:, 1]

        return agb_output, cc_output

def remove_freeze_attribute(model_path):
    with h5py.File(model_path, 'r+') as f:
        # Read the model configuration
        model_config = f.attrs['model_config']
        
        # Ensure the model_config is a string if it's not already
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')

        # Validate JSON before modification
        try:
            json.loads(model_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model_config before modification: {e}")


        # Replace the freeze attribute
        model_config = model_config.replace('"freeze": true,', '"freeze": True,')
        model_config = model_config.replace(', "freeze": true', ', "freeze": True')

        # Validate JSON after modification
        try:
            json.loads(model_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model_config after modification: {e}")
        
        # Write the updated configuration back
        f.attrs['model_config'] = model_config.encode('utf-8')

def load_model_with_hdf5():
    # Path to your model
    model_path = 'model/deepforest/final_model_4000_epochs_35.h5'

    # Remove the freeze attribute from the model configuration
    remove_freeze_attribute(model_path)

    # Now load the model
    model = tf.keras.models.load_model(model_path)

    # Summarize the model architecture
    model.summary()

    return model



# Function to load and modify the DeepForest model
def load_modified_deepforest_model():
    model = main.deepforest()
    model.use_release()

    # Replace the model's regression head with the MLP regression head
    modified_model = ModifiedDeepForestAGB_CC(model)
    
    return modified_model


if __name__ == '__main__':
    # load_model_with_hdf5()
    model = load_modified_deepforest_model()

    print(model)


    
