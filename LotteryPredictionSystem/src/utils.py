import logging
from datetime import datetime
import numpy as np

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'lottery_prediction_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def format_predictions(rf_pred, markov_pred, lstm_pred):
    """Format predictions from all models"""
    try:
        logging.info(f"Initial predictions - RF: {rf_pred}, Markov: {markov_pred}, LSTM: {lstm_pred}")

        # Convert predictions to numpy arrays and ensure they are 1D
        rf_pred = np.array(rf_pred).ravel()
        markov_pred = np.array(markov_pred).ravel()
        lstm_pred = np.array(lstm_pred).ravel()

        # Take first 3 predictions from each model
        rf_pred = rf_pred[:3]
        markov_pred = markov_pred[:3]
        lstm_pred = lstm_pred[:3]

        # Ensure all predictions are within valid range (1-80)
        rf_pred = np.clip(rf_pred, 1, 80)
        markov_pred = np.clip(markov_pred, 1, 80)
        lstm_pred = np.clip(lstm_pred, 1, 80)

        # Convert to integer type
        rf_pred = rf_pred.astype(int)
        markov_pred = markov_pred.astype(int)
        lstm_pred = lstm_pred.astype(int)

        formatted_predictions = {
            'random_forest_xgboost': rf_pred.tolist(),
            'markov_chain_xgboost': markov_pred.tolist(),
            'lstm': lstm_pred.tolist()
        }

        logging.info("Formatted predictions:")
        for model, pred in formatted_predictions.items():
            logging.info(f"{model}: {pred}")

        return formatted_predictions
    except Exception as e:
        logging.error(f"Error formatting predictions: {str(e)}")
        raise RuntimeError(f"Failed to format predictions: {str(e)}")

def validate_predictions(predictions):
    """Validate prediction results"""
    try:
        if not isinstance(predictions, dict):
            logging.error(f"Invalid predictions type: {type(predictions)}")
            return False

        for model, pred in predictions.items():
            # Log the prediction details for debugging
            logging.info(f"Validating {model} predictions: {pred}")

            # Check if prediction exists and has correct length
            if not pred or len(pred) != 3:
                logging.error(f"Invalid predictions from {model}: wrong length {len(pred) if pred else 0}")
                return False

            # Check if predictions are within valid range (1-80)
            if not all(isinstance(num, (int, np.integer)) and 1 <= num <= 80 for num in pred):
                invalid_nums = [num for num in pred if not (isinstance(num, (int, np.integer)) and 1 <= num <= 80)]
                logging.error(f"Invalid numbers in predictions from {model}: {invalid_nums}")
                return False

            # Check for duplicates
            if len(set(pred)) != len(pred):
                logging.error(f"Duplicate numbers in predictions from {model}: {pred}")
                return False

            logging.info(f"Validated predictions from {model}: {pred}")

        return True
    except Exception as e:
        logging.error(f"Error validating predictions: {str(e)}")
        return False