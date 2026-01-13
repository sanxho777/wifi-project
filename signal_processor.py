"""
Signal Processor Module
Preprocesses raw CSI data for AI model input.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import List, Tuple
from collections import deque


class SignalProcessor:
    """
    Processes raw CSI data into features for AI model.
    Includes: filtering, feature extraction, normalization.
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_antennas = config['hardware']['num_antennas']
        self.num_subcarriers = config['hardware']['num_subcarriers']
        self.sampling_rate = config['hardware']['sampling_rate']
        self.input_timesteps = config['model']['input_timesteps']

        # Buffer to store recent CSI data
        self.csi_buffer = deque(maxlen=self.input_timesteps)

        # Design filters
        self._design_filters()

        print(f"[Signal Processor] Initialized")
        print(f"  Input timesteps: {self.input_timesteps}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")

    def _design_filters(self):
        """Design signal processing filters."""
        # Butterworth bandpass filter for noise reduction
        # Remove DC component and high-frequency noise
        nyquist = self.sampling_rate / 2
        low_cutoff = 0.5 / nyquist   # Remove DC drift
        high_cutoff = 20 / nyquist    # Remove high-freq noise

        self.b_filter, self.a_filter = signal.butter(
            4, [low_cutoff, high_cutoff], btype='band'
        )

    def add_csi_sample(self, csi_data: np.ndarray):
        """
        Add a new CSI sample to the buffer.
        Args:
            csi_data: Shape (num_antennas, num_subcarriers, 2)
        """
        self.csi_buffer.append(csi_data)

    def is_ready(self) -> bool:
        """Check if buffer has enough data for processing."""
        return len(self.csi_buffer) >= self.input_timesteps

    def process(self) -> np.ndarray:
        """
        Process buffered CSI data into model input.
        Returns: Processed features ready for AI model
        """
        if not self.is_ready():
            raise ValueError("Not enough data in buffer")

        # Convert buffer to numpy array
        # Shape: (timesteps, antennas, subcarriers, 2)
        csi_sequence = np.array(list(self.csi_buffer))

        # Extract features
        features = []

        # 1. Amplitude features
        amplitude_features = self._extract_amplitude_features(csi_sequence)
        features.append(amplitude_features)

        # 2. Phase features
        phase_features = self._extract_phase_features(csi_sequence)
        features.append(phase_features)

        # 3. Doppler features (frequency domain)
        doppler_features = self._extract_doppler_features(csi_sequence)
        features.append(doppler_features)

        # 4. Spatial correlation features
        spatial_features = self._extract_spatial_features(csi_sequence)
        features.append(spatial_features)

        # Concatenate all features
        processed = np.concatenate(features, axis=-1)

        # Normalize
        processed = self._normalize(processed)

        return processed

    def _extract_amplitude_features(self, csi_sequence: np.ndarray) -> np.ndarray:
        """
        Extract amplitude-based features.
        CSI amplitude reflects signal strength and attenuation.
        """
        # Get amplitude (first channel)
        amplitude = csi_sequence[:, :, :, 0]  # Shape: (time, antennas, subcarriers)

        # Apply filtering to remove noise
        filtered_amp = np.zeros_like(amplitude)
        for ant in range(self.num_antennas):
            for sub in range(self.num_subcarriers):
                filtered_amp[:, ant, sub] = signal.filtfilt(
                    self.b_filter, self.a_filter, amplitude[:, ant, sub]
                )

        # Calculate amplitude variance (motion indicator)
        amp_variance = np.var(filtered_amp, axis=0, keepdims=True)
        amp_variance = np.repeat(amp_variance, self.input_timesteps, axis=0)

        # Stack original and variance
        features = np.stack([filtered_amp, amp_variance], axis=-1)

        return features  # Shape: (time, antennas, subcarriers, 2)

    def _extract_phase_features(self, csi_sequence: np.ndarray) -> np.ndarray:
        """
        Extract phase-based features.
        Phase changes indicate movement and location.
        """
        # Get phase (second channel)
        phase = csi_sequence[:, :, :, 1]  # Shape: (time, antennas, subcarriers)

        # Unwrap phase to handle 2π discontinuities
        unwrapped_phase = np.unwrap(phase, axis=0)

        # Calculate phase difference (velocity indicator)
        phase_diff = np.diff(unwrapped_phase, axis=0, prepend=unwrapped_phase[0:1])

        # Sanitize phase (remove outliers)
        sanitized_phase = self._sanitize_phase(phase)

        # Stack phase and phase difference
        features = np.stack([sanitized_phase, phase_diff], axis=-1)

        return features  # Shape: (time, antennas, subcarriers, 2)

    def _extract_doppler_features(self, csi_sequence: np.ndarray) -> np.ndarray:
        """
        Extract Doppler shift features (frequency domain).
        Doppler shift indicates movement speed and direction.
        """
        amplitude = csi_sequence[:, :, :, 0]

        # Compute FFT along time axis for each antenna-subcarrier pair
        doppler_spectrum = np.zeros((self.input_timesteps, self.num_antennas,
                                     self.num_subcarriers, 1))

        for ant in range(self.num_antennas):
            for sub in range(self.num_subcarriers):
                # FFT of amplitude over time
                fft_result = fft(amplitude[:, ant, sub])
                power_spectrum = np.abs(fft_result)

                # Store magnitude
                doppler_spectrum[:, ant, sub, 0] = power_spectrum

        return doppler_spectrum

    def _extract_spatial_features(self, csi_sequence: np.ndarray) -> np.ndarray:
        """
        Extract spatial correlation features across antennas.
        Spatial patterns help locate objects.
        """
        features = []

        # For each time step
        for t in range(self.input_timesteps):
            csi_t = csi_sequence[t]  # Shape: (antennas, subcarriers, 2)

            # Calculate correlation between antenna pairs
            correlations = []
            for i in range(self.num_antennas):
                for j in range(i + 1, self.num_antennas):
                    # Flatten subcarriers for each antenna
                    sig_i = csi_t[i, :, 0]  # Amplitude
                    sig_j = csi_t[j, :, 0]

                    # Correlation coefficient
                    corr = np.corrcoef(sig_i, sig_j)[0, 1]
                    correlations.append(corr)

            # Pad to fixed size
            while len(correlations) < self.num_subcarriers:
                correlations.append(0.0)

            features.append(correlations[:self.num_subcarriers])

        features = np.array(features)  # Shape: (time, subcarriers)

        # Reshape to match other feature dimensions
        features = features[:, np.newaxis, :, np.newaxis]  # (time, 1, subcarriers, 1)

        # Broadcast to all antennas
        features = np.repeat(features, self.num_antennas, axis=1)

        return features

    def _sanitize_phase(self, phase: np.ndarray) -> np.ndarray:
        """Remove phase outliers and normalize to [-π, π]."""
        sanitized = np.copy(phase)

        # Wrap to [-π, π]
        sanitized = np.arctan2(np.sin(sanitized), np.cos(sanitized))

        # Remove outliers (values beyond 3 standard deviations)
        mean = np.mean(sanitized)
        std = np.std(sanitized)
        sanitized = np.clip(sanitized, mean - 3*std, mean + 3*std)

        return sanitized

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.
        Important for neural network training.
        """
        # Min-max normalization
        data_min = np.min(data)
        data_max = np.max(data)

        if data_max - data_min < 1e-6:
            return data

        normalized = (data - data_min) / (data_max - data_min)

        return normalized

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get the shape of processed features."""
        if not self.is_ready():
            # Estimate shape
            # 4 feature types, each with different channels
            # Amplitude: 2 channels, Phase: 2 channels, Doppler: 1 channel, Spatial: 1 channel
            total_channels = 2 + 2 + 1 + 1  # = 6 channels

            return (self.input_timesteps, self.num_antennas,
                   self.num_subcarriers, total_channels)
        else:
            # Process and get actual shape
            features = self.process()
            return features.shape


if __name__ == "__main__":
    # Test the processor
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processor = SignalProcessor(config)

    print("Testing signal processor...")

    # Generate some test CSI data
    for i in range(config['model']['input_timesteps']):
        test_csi = np.random.randn(
            config['hardware']['num_antennas'],
            config['hardware']['num_subcarriers'],
            2
        )
        processor.add_csi_sample(test_csi)

    if processor.is_ready():
        features = processor.process()
        print(f"Processed features shape: {features.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print("Signal processor test PASSED!")
