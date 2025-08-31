"""
Data preprocessing module for Airbnb price prediction.
Handles data cleaning, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Class for preprocessing Airbnb data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="median")

    def load_data(self, file_path):
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self, df):
        """Clean the dataset by handling missing values and duplicates."""
        print("Cleaning data...")

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values
        missing_before = df.isnull().sum().sum()

        # Fill missing values in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

        # Fill missing values in categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

        missing_after = df.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values")

        return df

    def engineer_features(self, df):
        """Create new features from existing data."""
        print("Engineering features...")

        # Create total rooms feature
        if all(col in df.columns for col in ["bedrooms", "bathrooms", "studios"]):
            df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["studios"]

        # Create price per guest feature
        if all(col in df.columns for col in ["price", "guests"]):
            df["price_per_guest"] = df["price"] / df["guests"]

        # Create price per room feature
        if "total_rooms" in df.columns and "price" in df.columns:
            df["price_per_room"] = df["price"] / df["total_rooms"]

        # Extract amenities features
        if "amenities" in df.columns:
            df = self._extract_amenities_features(df)

        # Extract features from features column
        if "features" in df.columns:
            df = self._extract_features_info(df)

        # Create rating categories
        if "rating" in df.columns:
            df["rating_category"] = pd.cut(
                df["rating"], bins=[0, 4.0, 4.5, 5.0], labels=["Low", "Medium", "High"]
            )

        # Create review categories
        if "reviews" in df.columns:
            df["review_category"] = pd.cut(
                df["reviews"],
                bins=[0, 10, 50, 100, float("inf")],
                labels=["Very Few", "Few", "Moderate", "Many"],
            )

        print("Feature engineering completed!")
        return df

    def _extract_amenities_features(self, df):
        """Extract boolean features from amenities column."""
        # Common amenities to look for
        common_amenities = [
            "wifi",
            "kitchen",
            "air conditioning",
            "parking",
            "pool",
            "gym",
            "balcony",
            "garden",
            "terrace",
            "fireplace",
        ]

        for amenity in common_amenities:
            col_name = f'has_{amenity.replace(" ", "_")}'
            df[col_name] = df["amenities"].str.contains(amenity, case=False, na=False)

        return df

    def _extract_features_info(self, df):
        """Extract information from features column."""
        # Extract guest count
        df["guests_from_features"] = (
            df["features"].str.extract(r"(\d+)\s*guests?").astype(float)
        )

        # Extract view types
        df["has_mountain_view"] = df["features"].str.contains(
            "mountain view", case=False, na=False
        )
        df["has_valley_view"] = df["features"].str.contains(
            "valley view", case=False, na=False
        )

        return df

    def prepare_for_modeling(self, df, target_col="price"):
        """Prepare data for machine learning modeling."""
        print("Preparing data for modeling...")

        # Select features
        feature_cols = self._select_features(df)
        X = df[feature_cols].copy()
        y = df[target_col]

        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

        # Encode categorical variables
        X = self._encode_categorical(X)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Data prepared! Training set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

    def _select_features(self, df):
        """Select relevant features for modeling."""
        # Base features
        base_features = ["rating", "reviews", "bathrooms", "beds", "guests", "bedrooms"]

        # Add engineered features if they exist
        engineered_features = ["total_rooms", "price_per_guest", "price_per_room"]
        available_engineered = [f for f in engineered_features if f in df.columns]

        # Add boolean features
        boolean_features = [col for col in df.columns if col.startswith("has_")]

        # Add categorical features
        categorical_features = ["country", "checkin", "checkout"]
        available_categorical = [f for f in categorical_features if f in df.columns]

        all_features = (
            base_features
            + available_engineered
            + boolean_features
            + available_categorical
        )
        return [f for f in all_features if f in df.columns]

    def _encode_categorical(self, X):
        """Encode categorical variables."""
        categorical_cols = X.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        return X

    def save_preprocessed_data(self, df, file_path):
        """Save preprocessed data to CSV."""
        df.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")


def main():
    """Main function for testing the preprocessor."""
    preprocessor = DataPreprocessor()

    # Example usage
    print("DataPreprocessor class created successfully!")
    print("Use this class to preprocess your Airbnb dataset.")


if __name__ == "__main__":
    main()

