from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import logging

class RandomForestModel:
    def __init__(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        logging.info("RandomForest and XGBoost models initialized")

    def train(self, X_train, y_train):
        """Train both Random Forest and XGBoost models"""
        try:
            if X_train is None or y_train is None:
                raise ValueError("Training data cannot be None")

            self.rf_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            logging.info("Random Forest and XGBoost models trained successfully")
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise RuntimeError(f"Failed to train models: {str(e)}")

    def predict(self, X_test):
        """Make predictions using ensemble of both models"""
        try:
            if X_test is None:
                raise ValueError("Test data cannot be None")

            # Get predictions from both models
            rf_pred = self.rf_model.predict(X_test).ravel()
            xgb_pred = self.xgb_model.predict(X_test).ravel()

            # Combine predictions
            ensemble_pred = (rf_pred + xgb_pred) / 2

            # Create probability distribution for numbers 1-80
            number_probs = np.zeros(80)
            for i, prob in enumerate(ensemble_pred):
                idx = min(int(prob % 80), 79)  # Ensure index is within bounds
                number_probs[idx] += prob

            # Get top 3 most likely numbers (adding 1 since we want 1-80 range)
            top_numbers = np.argsort(number_probs)[-3:][::-1] + 1

            # Ensure predictions are unique and within valid range
            top_numbers = np.clip(top_numbers, 1, 80)

            # If we have duplicates, replace them with next highest probability numbers
            unique_numbers = set()
            final_numbers = []
            prob_indices = np.argsort(number_probs)[::-1]

            for idx in prob_indices:
                num = idx + 1
                if num not in unique_numbers and 1 <= num <= 80:
                    unique_numbers.add(num)
                    final_numbers.append(num)
                    if len(final_numbers) == 3:
                        break

            final_numbers = np.array(final_numbers)
            logging.info(f"Random Forest final predictions: {final_numbers}")
            return final_numbers

        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            if X_test is None or y_test is None:
                raise ValueError("Test data cannot be None")

            rf_score = self.rf_model.score(X_test, y_test)
            xgb_score = self.xgb_model.score(X_test, y_test)

            return {
                'random_forest_score': rf_score,
                'xgboost_score': xgb_score,
                'ensemble_score': (rf_score + xgb_score) / 2
            }
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise RuntimeError(f"Failed to evaluate models: {str(e)}")