"""
Model training module for Airbnb price prediction.
Trains multiple machine learning models and saves them.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ModelTrainer:
    """Class for training multiple ML models."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and evaluate them."""
        print("Training multiple models...")

        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=5),
        }

        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results[name] = {
                    "model": model,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                }

                self.models[name] = model

                print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")

                # Update best model
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = name

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        # Display results
        self._display_results(results)
        return results

    def _display_results(self, results):
        """Display training results."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING RESULTS")
        print("=" * 60)

        results_df = pd.DataFrame(results).T.drop("model", axis=1)
        display(results_df.round(4))

        print(f"\nBest performing model: {self.best_model} (R²: {self.best_score:.4f})")

    def hyperparameter_tuning(self, X_train, y_train, model_name="Random Forest"):
        """Perform hyperparameter tuning for a specific model."""
        print(f"Performing hyperparameter tuning for {model_name}...")

        if model_name == "Random Forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            model = RandomForestRegressor(random_state=42)

        elif model_name == "XGBoost":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            }
            model = xgb.XGBRegressor(random_state=42)

        elif model_name == "LightGBM":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 62, 127],
            }
            model = lgb.LGBMRegressor(random_state=42)

        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None

        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        # Update best model
        self.models[f"{model_name} (Tuned)"] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def save_models(self, output_dir="../models/"):
        """Save trained models to disk."""
        print("Saving models...")

        for name, model in self.models.items():
            try:
                # Clean filename
                filename = (
                    name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                )
                filepath = f"{output_dir}{filename}.pkl"

                # Save model
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

                print(f"Saved {name} to {filepath}")

            except Exception as e:
                print(f"Error saving {name}: {e}")

    def load_model(self, model_path):
        """Load a saved model from disk."""
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, model_name, X):
        """Make predictions using a specific model."""
        if model_name in self.models:
            return self.models[model_name].predict(X)
        else:
            print(f"Model {model_name} not found!")
            return None


def main():
    """Main function for testing the model trainer."""
    trainer = ModelTrainer()
    print("ModelTrainer class created successfully!")
    print("Use this class to train and save your ML models.")


if __name__ == "__main__":
    main()

