from src.data_collector import DataCollector
from src.data_preprocessor import DataPreprocessor
from src.models.random_forest_model import RandomForestModel
from src.models.markov_xgboost_model import MarkovXGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils import setup_logging, format_predictions, validate_predictions
import logging
import numpy as np
import traceback

def main():
    setup_logging()
    logging.info("Starting lottery prediction system")

    try:
        # Initialize components
        collector = DataCollector()
        preprocessor = DataPreprocessor()

        # Collect data
        logging.info("Fetching historical lottery data...")
        df = collector.fetch_data()
        if df is None:
            raise Exception("Failed to collect data")
        logging.info(f"Successfully collected {len(df)} records")

        # Log detailed data structure
        logging.info(f"Data columns: {df.columns.tolist()}")
        logging.info(f"Sample data:\n{df.head(1)}")

        if 'numbers' not in df.columns:
            raise Exception("Required 'numbers' column not found in data")

        # Validate numbers format
        sample_numbers = df['numbers'].iloc[0]
        logging.info(f"Sample numbers format: {type(sample_numbers)}, Value: {sample_numbers}")

        # Preprocess data
        logging.info("Preprocessing data...")
        X_processed = preprocessor.prepare_data(df)

        if X_processed is None or any(x is None for x in X_processed):
            raise Exception("Failed to preprocess data")

        rf_features, markov_features, lstm_features = X_processed
        logging.info(f"RF features shape: {rf_features.shape if rf_features is not None else 'None'}")
        logging.info(f"Markov features shape: {markov_features.shape if markov_features is not None else 'None'}")

        if lstm_features is not None:
            lstm_seq, lstm_target = lstm_features
            logging.info(f"LSTM sequences shape: {lstm_seq.shape if lstm_seq is not None else 'None'}")
            logging.info(f"LSTM targets shape: {lstm_target.shape if lstm_target is not None else 'None'}")
        else:
            raise Exception("LSTM features are None")

        # Split data for each model
        logging.info("Splitting data...")
        rf_train, rf_test = preprocessor.split_data(rf_features)
        markov_train, markov_test = preprocessor.split_data(markov_features)

        # Handle LSTM data separately
        lstm_sequences, lstm_targets = lstm_features
        lstm_split = preprocessor.split_data(lstm_sequences, lstm_targets)

        if len(lstm_split) == 4:
            lstm_train_seq, lstm_train_target, lstm_test_seq, lstm_test_target = lstm_split
            logging.info("LSTM data split completed successfully")
        else:
            raise Exception("Invalid LSTM data split")

        # Train and predict with Random Forest/XGBoost
        logging.info("Training Random Forest model...")
        rf_model = RandomForestModel()
        rf_model.train(rf_train, rf_train)  # Using same data for X and y as it's a time series
        rf_predictions = rf_model.predict(rf_test[-1:])
        logging.info(f"RF Predictions: {rf_predictions}")

        # Train and predict with Markov Chain + XGBoost
        logging.info("Training Markov Chain + XGBoost model...")
        markov_model = MarkovXGBoostModel()
        markov_model.train(markov_train, markov_train, lstm_sequences)
        current_numbers = df['numbers'].iloc[-1]
        markov_predictions = markov_model.predict(current_numbers, markov_test[-1:])
        logging.info(f"Markov Predictions: {markov_predictions}")

        # Train and predict with LSTM
        logging.info("Training LSTM model...")
        lstm_model = LSTMModel(sequence_length=10, n_features=20)
        lstm_model.train(lstm_train_seq, lstm_train_target)
        lstm_predictions = lstm_model.predict(lstm_test_seq[-1:])
        logging.info(f"LSTM Predictions: {lstm_predictions}")

        # Format and validate predictions
        predictions = format_predictions(rf_predictions, markov_predictions, lstm_predictions)

        if not validate_predictions(predictions):
            raise Exception("Invalid predictions detected")

        # Print results
        logging.info("Predictions for next draw:")
        for model, pred in predictions.items():
            logging.info(f"{model}: {pred}")

        # Evaluate models
        logging.info("Evaluating model performance...")
        rf_eval = rf_model.evaluate(rf_test, rf_test)
        markov_eval = markov_model.evaluate(markov_test, markov_test)
        lstm_eval = lstm_model.evaluate(lstm_test_seq, lstm_test_target)

        logging.info("Model Evaluations:")
        logging.info(f"Random Forest/XGBoost: {rf_eval}")
        logging.info(f"Markov Chain + XGBoost: {markov_eval}")
        logging.info(f"LSTM: {lstm_eval}")

        return predictions

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()