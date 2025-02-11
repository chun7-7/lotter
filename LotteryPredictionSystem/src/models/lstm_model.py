import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import traceback

class LSTMModel:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build enhanced LSTM model architecture"""
        try:
            model = Sequential([
                # First LSTM layer with batch normalization
                LSTM(128, input_shape=(self.sequence_length, self.n_features), 
                     return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),

                # Second LSTM layer
                LSTM(64, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),

                # Third LSTM layer
                LSTM(32, return_sequences=False),
                BatchNormalization(),
                Dropout(0.2),

                # Dense layers
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dense(20)  # Output layer for 20 numbers
            ])

            model.compile(
                optimizer='adam', 
                loss='mse', 
                metrics=['mae'],
                run_eagerly=True
            )

            self.model = model
            logging.info("Enhanced LSTM model built successfully")
            logging.info(f"Model input shape: {self.sequence_length}x{self.n_features}")
            model.summary(print_fn=logging.info)
        except Exception as e:
            logging.error(f"Error building LSTM model: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to build LSTM model: {str(e)}")

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model with advanced training configurations"""
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")

            logging.info("Starting enhanced LSTM model training...")
            logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}")

            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )

            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            # Log detailed training results
            logging.info("LSTM model training completed")
            logging.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
            logging.info(f"Final training MAE: {history.history['mae'][-1]:.4f}")
            if 'val_loss' in history.history:
                logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
                logging.info(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")

            return history
        except Exception as e:
            logging.error(f"Error training LSTM model: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to train LSTM model: {str(e)}")

    def predict(self, X_test):
        """Make predictions using LSTM model"""
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")

            logging.info(f"Making predictions with input shape: {X_test.shape}")
            predictions = self.model.predict(X_test, verbose=1)
            logging.info(f"Raw predictions shape: {predictions.shape}")

            # Get top 3 most likely numbers
            top_numbers = np.argsort(predictions[0])[-3:][::-1] + 1
            logging.info(f"Top 3 predicted numbers: {top_numbers}")

            return top_numbers
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")

            logging.info(f"Evaluating model with test data shape: X={X_test.shape}, y={y_test.shape}")
            loss, mae = self.model.evaluate(X_test, y_test, verbose=1)

            metrics = {
                'loss': float(loss),
                'mae': float(mae)
            }
            logging.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"Error evaluating LSTM model: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to evaluate LSTM model: {str(e)}")