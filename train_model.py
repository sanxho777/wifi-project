"""
Model Training Script
Train the AI model with collected CSI data.
"""

import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import json
from datetime import datetime

from ai_model import WiFiSensingModel


class CSIDataset(Dataset):
    """PyTorch dataset for CSI training data."""

    def __init__(self, data_dir: str, class_names: list):
        self.samples = []
        self.labels = []
        self.class_names = class_names

        print(f"[Dataset] Loading data from {data_dir}")

        # Load all .npy files
        feature_files = glob.glob(os.path.join(data_dir, "*_features.npy"))

        for feature_file in feature_files:
            # Load features
            features = np.load(feature_file)

            # Load metadata to get label
            metadata_file = feature_file.replace('_features.npy', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    label_name = metadata['label']

                    # Convert label name to index
                    if label_name in class_names:
                        label_idx = class_names.index(label_name)

                        # Add all samples from this file
                        for sample in features:
                            self.samples.append(sample)
                            self.labels.append(label_idx)

                        print(f"  Loaded {len(features)} samples for '{label_name}'")
                    else:
                        print(f"  [Warning] Unknown label '{label_name}', skipping...")

        print(f"[Dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get sample and label
        features = self.samples[idx]
        label = self.labels[idx]

        # Convert to tensors
        features = torch.from_numpy(features).float()

        # Create multi-hot label vector
        label_vector = torch.zeros(len(self.class_names))
        label_vector[label] = 1.0

        # Dummy location (for now)
        # In real training, you'd have ground truth locations
        location = torch.zeros((len(self.class_names), 4))  # (x, y, z, confidence)

        return features, label_vector, location


def train_model(config_path: str, data_dir: str, epochs: int, save_dir: str):
    """
    Train the WiFi sensing model.

    Args:
        config_path: Path to configuration file
        data_dir: Directory containing training data
        epochs: Number of training epochs
        save_dir: Where to save trained model
    """
    print("=" * 70)
    print("WiFi Room Sensing - Model Training")
    print("=" * 70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print("\n[Training] Loading dataset...")
    dataset = CSIDataset(data_dir, config['model']['class_names'])

    if len(dataset) == 0:
        print("[ERROR] No training data found!")
        print(f"  Please run: python collect_training_data.py --label <object_name>")
        return

    # Split into train/validation
    val_split = config['training']['validation_split']
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"[Training] Train samples: {train_size}")
    print(f"[Training] Validation samples: {val_size}")

    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\n[Training] Creating model...")
    model_wrapper = WiFiSensingModel(config)
    model = model_wrapper.model

    # Loss functions
    criterion_class = nn.BCEWithLogitsLoss()  # Multi-label classification
    criterion_loc = nn.MSELoss()              # Location regression

    # Optimizer
    learning_rate = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n[Training] Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(epochs):
        epoch_start = datetime.now()

        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (features, labels, locations) in enumerate(train_loader):
            # Move to device
            features = features.to(model_wrapper.device)
            labels = labels.to(model_wrapper.device)
            locations = locations.to(model_wrapper.device)

            # Reshape features: (batch, time, antennas, subcarriers, channels)
            # to (batch, channels, time, antennas, subcarriers)
            features = features.permute(0, 4, 1, 2, 3)

            # Forward pass
            class_logits, loc_pred = model(features)

            # Compute losses
            loss_class = criterion_class(class_logits, labels)
            loss_loc = criterion_loc(loc_pred, locations)
            loss = loss_class + 0.5 * loss_loc

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for features, labels, locations in val_loader:
                features = features.to(model_wrapper.device)
                labels = labels.to(model_wrapper.device)
                locations = locations.to(model_wrapper.device)

                features = features.permute(0, 4, 1, 2, 3)

                class_logits, loc_pred = model(features)

                loss_class = criterion_class(class_logits, labels)
                loss_loc = criterion_loc(loc_pred, locations)
                loss = loss_class + 0.5 * loss_loc

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print epoch summary
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\n[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        print("=" * 70)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            model_wrapper.save_model(best_model_path)
            print(f"[Training] New best model saved! Val Loss: {avg_val_loss:.4f}\n")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            model_wrapper.save_model(checkpoint_path)

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    model_wrapper.save_model(final_model_path)

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"  - best_model.pth (use this for inference)")
    print(f"  - final_model.pth")
    print(f"  - training_history.json")

    print("\nTo use the trained model:")
    print(f"  1. Update config.yaml:")
    print(f"     model:")
    print(f"       pretrained_path: '{os.path.join(save_dir, 'best_model.pth')}'")
    print(f"  2. Run: python main.py")


def main():
    parser = argparse.ArgumentParser(
        description='Train WiFi room sensing model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='./data/training',
        help='Training data directory (default: ./data/training)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./models',
        help='Where to save models (default: ./models)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )

    args = parser.parse_args()

    train_model(
        config_path=args.config,
        data_dir=args.data,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
