"""
AI Model Module
Neural network for detecting objects from WiFi CSI signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import os


class WiFiSensingCNN(nn.Module):
    """
    Convolutional Neural Network for WiFi sensing.

    Architecture:
    1. 3D CNN layers to process spatial-temporal CSI data
    2. Feature extraction and pooling
    3. Classification head for object detection
    4. Localization head for 3D position estimation
    """

    def __init__(self, config: dict):
        super(WiFiSensingCNN, self).__init__()

        self.config = config
        self.num_classes = config['model']['num_classes']
        self.input_timesteps = config['model']['input_timesteps']
        self.num_antennas = config['hardware']['num_antennas']
        self.num_subcarriers = config['hardware']['num_subcarriers']

        # Input channels (from signal processor features)
        # Amplitude(2) + Phase(2) + Doppler(1) + Spatial(1) = 6 channels
        self.input_channels = 6

        # === FEATURE EXTRACTION LAYERS ===
        # Conv3D: (batch, channels, depth, height, width)
        # depth=time, height=antennas, width=subcarriers

        self.conv1 = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=32,
            kernel_size=(5, 3, 5),  # (time, antennas, subcarriers)
            padding=(2, 1, 2)
        )
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 2, 3),
            padding=(1, 0, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 2, 3),
            padding=(1, 0, 1)
        )
        self.bn3 = nn.BatchNorm3d(128)

        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=(2, 1, 2))

        # === CLASSIFICATION HEAD ===
        # Predict what objects are present
        # Calculate flattened size after convolutions
        self._calculate_flatten_size()

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        # Object classification (multi-label)
        self.fc_class = nn.Linear(256, self.num_classes)

        # === LOCALIZATION HEAD ===
        # Predict 3D positions of detected objects
        # Output: (x, y, z, confidence) for each object
        self.fc_loc = nn.Linear(256, self.num_classes * 4)

        print(f"[AI Model] WiFiSensingCNN initialized")
        print(f"  Input shape: ({self.input_channels}, {self.input_timesteps}, "
              f"{self.num_antennas}, {self.num_subcarriers})")
        print(f"  Number of classes: {self.num_classes}")

    def _calculate_flatten_size(self):
        """Calculate the size after convolution and pooling layers."""
        # Simulate a forward pass to get the size
        dummy_input = torch.zeros(
            1, self.input_channels, self.input_timesteps,
            self.num_antennas, self.num_subcarriers
        )

        x = F.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        self.flatten_size = x.view(1, -1).shape[1]

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, time, antennas, subcarriers)

        Returns:
            class_logits: Object classification scores (batch, num_classes)
            locations: 3D positions (batch, num_classes, 4) - (x, y, z, confidence)
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        features = self.dropout2(x)

        # Classification head
        class_logits = self.fc_class(features)

        # Localization head
        locations = self.fc_loc(features)
        locations = locations.view(-1, self.num_classes, 4)

        return class_logits, locations


class WiFiSensingModel:
    """
    High-level wrapper for the WiFi sensing model.
    Handles training, inference, and model management.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = WiFiSensingCNN(config).to(self.device)

        # Class names
        self.class_names = config['model']['class_names']
        self.confidence_threshold = config['model']['confidence_threshold']

        # Load pre-trained weights if available
        pretrained_path = config['model'].get('pretrained_path', '')
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_model(pretrained_path)
            print(f"[AI Model] Loaded pre-trained model from {pretrained_path}")
        else:
            print(f"[AI Model] Using untrained model (training required)")
            print(f"  For best results, collect training data and train the model")

        print(f"[AI Model] Running on device: {self.device}")

    def predict(self, features: np.ndarray) -> Tuple[List[str], List[Tuple[float, float, float]]]:
        """
        Make predictions from processed CSI features.

        Args:
            features: Processed CSI features from SignalProcessor

        Returns:
            detected_objects: List of detected object names
            locations: List of 3D positions (x, y, z) for each object
        """
        self.model.eval()

        with torch.no_grad():
            # Convert to tensor
            # Add batch dimension and move channels to correct position
            # Expected: (batch, channels, time, antennas, subcarriers)
            input_tensor = torch.from_numpy(features).float()
            input_tensor = input_tensor.permute(3, 0, 1, 2)  # (channels, time, antennas, subcarriers)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            class_logits, locations = self.model(input_tensor)

            # Get predictions
            class_probs = torch.sigmoid(class_logits).cpu().numpy()[0]
            locations = locations.cpu().numpy()[0]

        # Extract detected objects
        detected_objects = []
        detected_locations = []

        for i, prob in enumerate(class_probs):
            if prob > self.confidence_threshold:
                detected_objects.append(self.class_names[i])

                # Get location (x, y, z, confidence)
                loc = locations[i]
                x, y, z, conf = loc

                # Scale to room dimensions
                room_dims = self.config['room']['dimensions']
                x = x * room_dims[0]  # Scale to room length
                y = y * room_dims[1]  # Scale to room width
                z = z * room_dims[2]  # Scale to room height

                detected_locations.append((x, y, z))

        return detected_objects, detected_locations

    def train_step(self, features_batch: np.ndarray, labels_batch: np.ndarray,
                   locations_batch: np.ndarray, optimizer, criterion_class, criterion_loc):
        """
        Single training step.

        Args:
            features_batch: Batch of CSI features
            labels_batch: Ground truth object labels (multi-hot)
            locations_batch: Ground truth 3D locations
            optimizer: PyTorch optimizer
            criterion_class: Classification loss function
            criterion_loc: Localization loss function

        Returns:
            loss: Total loss value
        """
        self.model.train()

        # Convert to tensors
        inputs = torch.from_numpy(features_batch).float().to(self.device)
        labels = torch.from_numpy(labels_batch).float().to(self.device)
        locations_gt = torch.from_numpy(locations_batch).float().to(self.device)

        # Permute inputs to (batch, channels, time, antennas, subcarriers)
        inputs = inputs.permute(0, 4, 1, 2, 3)

        # Forward pass
        class_logits, locations_pred = self.model(inputs)

        # Compute losses
        loss_class = criterion_class(class_logits, labels)
        loss_loc = criterion_loc(locations_pred, locations_gt)

        # Combined loss
        loss = loss_class + 0.5 * loss_loc

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
        print(f"[AI Model] Saved model to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"[AI Model] Loaded model from {filepath}")

    def get_model_info(self) -> Dict:
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }


if __name__ == "__main__":
    # Test the model
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing AI model...")

    # Create model
    ai_model = WiFiSensingModel(config)

    # Print model info
    info = ai_model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Device: {info['device']}")

    # Test prediction with dummy data
    dummy_features = np.random.randn(100, 3, 30, 6)  # (time, antennas, subcarriers, channels)

    objects, locations = ai_model.predict(dummy_features)

    print(f"\nTest prediction:")
    print(f"  Detected objects: {objects}")
    print(f"  Locations: {locations}")

    print("\nAI model test PASSED!")
