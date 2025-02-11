import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        logging.info("DataPreprocessor initialized")

    def prepare_data(self, df, sequence_length=10):
        """Prepare data for different models"""
        try:
            logging.info("Starting data preprocessing...")

            # Validate input data
            if df is None or df.empty:
                raise ValueError("Input DataFrame is None or empty")

            # Ensure numbers column exists
            if 'numbers' not in df.columns:
                raise ValueError("Numbers column not found in DataFrame")

            # Convert numbers to numpy arrays if they're lists or other formats
            try:
                numbers = np.array([
                    n if isinstance(n, np.ndarray) 
                    else np.array(n) if isinstance(n, (list, tuple)) 
                    else np.array(eval(n)) if isinstance(n, str) 
                    else None
                    for n in df['numbers']
                ])

                if numbers.dtype == object or None in numbers:
                    raise ValueError("Invalid number format in data")

                if numbers.shape[1] != 20:
                    raise ValueError(f"Invalid number of lottery numbers: {numbers.shape[1]}")

                logging.info(f"Successfully converted numbers to shape: {numbers.shape}")
            except Exception as e:
                raise ValueError(f"Error converting numbers: {str(e)}")

            # Create features for different models
            rf_features = self._create_rf_features(numbers)
            markov_features = self._create_markov_features(numbers)
            lstm_features = self._create_lstm_features(numbers, sequence_length)

            if all(v is not None for v in [rf_features, markov_features, lstm_features]):
                logging.info("Data preprocessing completed successfully")
                return rf_features, markov_features, lstm_features
            else:
                missing = []
                if rf_features is None: missing.append("RF")
                if markov_features is None: missing.append("Markov")
                if lstm_features is None: missing.append("LSTM")
                raise ValueError(f"Failed to create features for: {', '.join(missing)}")

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            return None, None, None

    def _create_rf_features(self, numbers):
        """Create features for Random Forest model"""
        try:
            features = pd.DataFrame()

            # Statistical features
            features['mean'] = numbers.mean(axis=1)
            features['std'] = numbers.std(axis=1)
            features['median'] = np.median(numbers, axis=1)
            features['min'] = numbers.min(axis=1)
            features['max'] = numbers.max(axis=1)

            # Frequency-based features
            for i in range(1, 81):
                features[f'freq_{i}'] = np.sum(numbers == i, axis=1)

            logging.info(f"Created Random Forest features with shape: {features.shape}")
            return features
        except Exception as e:
            logging.error(f"Error creating RF features: {str(e)}")
            return None

    def _create_markov_features(self, numbers):
        """Create features for Markov Chain model"""
        try:
            features = pd.DataFrame()

            # Create transition features
            for i in range(1, 81):
                features[f'prev_{i}'] = np.roll(np.sum(numbers == i, axis=1), 1)

            # Set first row to 0 as it has no previous state
            features.iloc[0] = 0

            logging.info(f"Created Markov Chain features with shape: {features.shape}")
            return features
        except Exception as e:
            logging.error(f"Error creating Markov features: {str(e)}")
            return None

    def _create_lstm_features(self, numbers, sequence_length):
        """Create sequences for LSTM model"""
        try:
            if len(numbers) <= sequence_length:
                raise ValueError(f"Not enough data for LSTM sequences: {len(numbers)} ≤ {sequence_length}")

            sequences = []
            targets = []

            for i in range(len(numbers) - sequence_length):
                seq = numbers[i:i+sequence_length]
                target = numbers[i+sequence_length]
                sequences.append(seq)
                targets.append(target)

            sequences = np.array(sequences)
            targets = np.array(targets)

            if len(sequences) > 0:
                logging.info(f"Created LSTM sequences with shape: {sequences.shape}")
                return sequences, targets
            else:
                raise ValueError("No sequences created for LSTM")

        except Exception as e:
            logging.error(f"Error creating LSTM features: {str(e)}")
            return None

    def split_data(self, features, targets=None, test_size=0.2):
        """Split data into training and testing sets"""
        try:
            if features is None:
                raise ValueError("Features cannot be None")

            split_idx = int(len(features) * (1 - test_size))

            if targets is not None:
                if len(features) != len(targets):
                    raise ValueError(f"Features and targets must have the same length: {len(features)} != {len(targets)}")

                return (
                    features[:split_idx], targets[:split_idx],
                    features[split_idx:], targets[split_idx:]
                )

            return features[:split_idx], features[split_idx:]

        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            return None, None if targets is None else None, None, None, None