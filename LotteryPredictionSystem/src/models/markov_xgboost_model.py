import numpy as np
import xgboost as xgb
from collections import defaultdict
import logging

class MarkovXGBoostModel:
    def __init__(self):
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        logging.info("MarkovXGBoost model initialized")

    def _build_transition_matrix(self, sequences):
        """Build Markov Chain transition matrix"""
        if sequences is None or len(sequences) == 0:
            raise ValueError("Sequences cannot be None or empty")

        for sequence in sequences:
            for i in range(len(sequence)-1):
                current_state = tuple(sequence[i])
                next_state = tuple(sequence[i+1])

                self.transition_matrix[current_state][next_state] += 1

        # Normalize probabilities
        for state in self.transition_matrix:
            total = sum(self.transition_matrix[state].values())
            if total > 0:  # Avoid division by zero
                for next_state in self.transition_matrix[state]:
                    self.transition_matrix[state][next_state] /= total

    def train(self, X_train, y_train, sequences):
        """Train both Markov Chain and XGBoost models"""
        try:
            if X_train is None or y_train is None:
                raise ValueError("Training data cannot be None")

            # Build Markov Chain
            self._build_transition_matrix(sequences)

            # Train XGBoost
            self.xgb_model.fit(X_train, y_train)

            logging.info("Markov Chain and XGBoost models trained successfully")
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise RuntimeError(f"Failed to train models: {str(e)}")

    def predict(self, current_state, X_test):
        """Make predictions using both models"""
        try:
            if current_state is None or X_test is None:
                raise ValueError("Input data cannot be None")

            # Markov Chain prediction
            current_state = tuple(current_state)
            markov_probs = self.transition_matrix[current_state]

            # XGBoost prediction
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_pred = xgb_pred.reshape(-1)  # Ensure 1D array

            # Combine predictions
            combined_pred = np.zeros(80)  # For numbers 1-80

            # Add Markov Chain probabilities
            for state, prob in markov_probs.items():
                for num in state:
                    combined_pred[num-1] += prob / len(state)  # Distribute probability evenly

            # Add XGBoost predictions with reshaping
            if len(xgb_pred) == 80:
                combined_pred += xgb_pred
            else:
                logging.warning(f"Unexpected XGBoost prediction shape: {xgb_pred.shape}")
                # Handle unexpected shape by using only Markov predictions

            # Get top 3 numbers
            top_numbers = np.argsort(combined_pred)[-3:][::-1] + 1

            return top_numbers
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            if X_test is None or y_test is None:
                raise ValueError("Test data cannot be None")

            xgb_score = self.xgb_model.score(X_test, y_test)
            return {
                'xgboost_score': xgb_score,
                'markov_chain_states': len(self.transition_matrix)
            }
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise RuntimeError(f"Failed to evaluate models: {str(e)}")