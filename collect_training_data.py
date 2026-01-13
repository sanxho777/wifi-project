"""
Training Data Collection Script
Collects CSI data with labels for training the AI model.
"""

import yaml
import argparse
import time
import numpy as np
import os
from datetime import datetime
import json

from csi_collector import CSICollector
from signal_processor import SignalProcessor


def collect_training_data(config_path: str, label: str, duration: int, output_dir: str):
    """
    Collect CSI data with a specific label.

    Args:
        config_path: Path to configuration file
        label: Label for this data (e.g., 'person', 'empty_room')
        duration: How long to collect (seconds)
        output_dir: Where to save data
    """
    print("=" * 70)
    print("WiFi Room Sensing - Training Data Collection")
    print("=" * 70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize collector and processor
    collector = CSICollector(config)
    processor = SignalProcessor(config)

    print(f"\nCollecting data for label: '{label}'")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_dir}")

    # Start collection
    if not collector.start_collection():
        print("[ERROR] Failed to start CSI collection!")
        return

    print("\n[INFO] Collection started!")
    print("      Make sure the labeled object/scenario is present in the room.")
    print("      Press Ctrl+C to stop early.\n")

    # Storage
    samples = []
    metadata = {
        'label': label,
        'start_time': datetime.now().isoformat(),
        'config': config,
        'samples': []
    }

    start_time = time.time()
    sample_count = 0

    try:
        while (time.time() - start_time) < duration:
            # Get CSI data
            csi_data = collector.get_csi_data(timeout=0.1)

            if csi_data is not None:
                # Add to processor
                processor.add_csi_sample(csi_data)

                # When buffer is ready, save processed features
                if processor.is_ready():
                    features = processor.process()

                    # Save sample
                    samples.append(features)
                    sample_count += 1

                    # Progress
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"\r[Progress] Samples: {sample_count} | "
                          f"Time: {elapsed:.1f}s / {duration}s | "
                          f"Remaining: {remaining:.1f}s", end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n[INFO] Collection stopped by user.")

    finally:
        collector.stop_collection()

    # Save data
    print(f"\n\n[INFO] Saving {len(samples)} samples...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}"

    # Save features as numpy array
    features_path = os.path.join(output_dir, f"{filename}_features.npy")
    np.save(features_path, np.array(samples))

    # Save metadata
    metadata['end_time'] = datetime.now().isoformat()
    metadata['num_samples'] = len(samples)
    metadata_path = os.path.join(output_dir, f"{filename}_metadata.json")

    with open(metadata_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        config_serializable = json.loads(json.dumps(config, default=str))
        metadata['config'] = config_serializable
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Data saved:")
    print(f"  Features: {features_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Total samples: {len(samples)}")

    print("\n" + "=" * 70)
    print("Collection complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Collect training data for WiFi room sensing'
    )
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help='Label for this data (e.g., person, chair, empty_room)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Collection duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/training',
        help='Output directory (default: ./data/training)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )

    args = parser.parse_args()

    collect_training_data(
        config_path=args.config,
        label=args.label,
        duration=args.duration,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
