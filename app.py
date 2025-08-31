"""
Streamlit app for Airbnb Price Prediction Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.data_preprocessing import DataPreprocessor
from src.model import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import *

# Page configuration
st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")


def main():
    st.title("🏠 Airbnb Price Prediction & Analysis")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "Home",
            "Data Upload",
            "Data Analysis",
            "Model Training",
            "Price Prediction",
            "About",
        ],
    )

    if page == "Home":
        show_home()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Model Training":
        show_model_training()
    elif page == "Price Prediction":
        show_price_prediction()
    elif page == "About":
        show_about()


def show_home():
    st.header("Welcome to Airbnb Price Prediction!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### What this app does:
        - **Analyzes** Airbnb listing data to understand pricing factors
        - **Trains** machine learning models to predict prices
        - **Identifies** key features influencing pricing
        - **Provides** interactive visualizations and insights
        
        ### Key Features:
        - 📊 Comprehensive data exploration
        - 🔧 Automated feature engineering
        - 🤖 Multiple ML algorithms
        - 📈 Performance evaluation
        - 🎯 Price prediction interface
        """
        )

    with col2:
        st.markdown(
            """
        ### Dataset Requirements:
        Your CSV should include:
        - `price` - Target variable
        - `rating` - Property rating
        - `reviews` - Number of reviews
        - `bathrooms`, `bedrooms`, `guests`
        - `country`, `address`
        - `amenities`, `features`
        
        ### Getting Started:
        1. Upload your dataset
        2. Explore the data
        3. Train models
        4. Make predictions
        """
        )

    st.markdown("---")
    st.info(
        "💡 **Tip**: Start by uploading your Airbnb dataset in the Data Upload section!"
    )


def show_data_upload():
    st.header("📁 Data Upload")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your Airbnb dataset in CSV format",
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")

            # Save to session state
            st.session_state["df"] = df
            st.session_state["uploaded_file"] = uploaded_file.name

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Basic info
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric(
                    "Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                )

            # Check for required columns
            required_cols = ["price", "rating", "reviews"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.warning(f"⚠️ Missing recommended columns: {missing_cols}")
            else:
                st.success("✅ All recommended columns found!")

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")


def show_data_analysis():
    st.header("📊 Data Analysis")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return

    df = st.session_state["df"]

    # Analysis options
    analysis_type = st.selectbox(
        "Choose analysis type:",
        [
            "Quick Overview",
            "Price Analysis",
            "Feature Analysis",
            "Correlation Analysis",
            "Interactive Dashboard",
        ],
    )

    if analysis_type == "Quick Overview":
        st.subheader("Quick Data Overview")
        quick_data_overview(df)

    elif analysis_type == "Price Analysis":
        st.subheader("Price Distribution Analysis")
        price_col = st.selectbox(
            "Select price column:",
            [col for col in df.columns if "price" in col.lower()],
        )
        if price_col:
            stats, outliers = analyze_price_distribution(df, price_col)
            st.write("**Price Statistics:**")
            st.write(stats)

            if len(outliers) > 0:
                st.write(f"**Outliers detected:** {len(outliers)} rows")
                st.dataframe(outliers.head())

    elif analysis_type == "Feature Analysis":
        st.subheader("Feature Analysis")
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            analyze_categorical_features(df, categorical_cols)

    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        create_correlation_heatmap(df)

    elif analysis_type == "Interactive Dashboard":
        st.subheader("Interactive Dashboard")
        price_col = st.selectbox(
            "Select price column for dashboard:",
            [col for col in df.columns if "price" in col.lower()],
        )
        if price_col:
            create_interactive_dashboard(df, price_col)


def show_model_training():
    st.header("🤖 Model Training")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return

    df = st.session_state["df"]

    # Preprocessing
    if st.button("🚀 Start Model Training"):
        with st.spinner("Training models..."):
            try:
                # Initialize preprocessor
                preprocessor = DataPreprocessor()

                # Clean and engineer features
                df_clean = preprocessor.clean_data(df.copy())
                df_engineered = preprocessor.engineer_features(df_clean)

                # Prepare for modeling
                X_train_scaled, X_test_scaled, y_train, y_test, feature_cols = (
                    preprocessor.prepare_for_modeling(df_engineered)
                )

                # Train models
                trainer = ModelTrainer()
                results = trainer.train_models(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )

                # Save results to session state
                st.session_state["trainer"] = trainer
                st.session_state["results"] = results
                st.session_state["feature_cols"] = feature_cols

                st.success("✅ Model training completed!")

                # Display results
                st.subheader("Training Results")
                results_df = pd.DataFrame(results).T.drop("model", axis=1)
                st.dataframe(results_df.round(4))

                # Save models
                if st.button("💾 Save Models"):
                    trainer.save_models()
                    st.success("✅ Models saved successfully!")

            except Exception as e:
                st.error(f"❌ Error during training: {e}")


def show_price_prediction():
    st.header("🎯 Price Prediction")

    if "trainer" not in st.session_state:
        st.warning("⚠️ Please train models first!")
        return

    trainer = st.session_state["trainer"]
    feature_cols = st.session_state.get("feature_cols", [])

    st.subheader("Make a Prediction")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
        reviews = st.number_input("Number of Reviews", 0, 1000, 50)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        beds = st.number_input("Beds", 1, 20, 2)

    with col2:
        guests = st.number_input("Guests", 1, 20, 4)
        bedrooms = st.number_input("Bedrooms", 1, 10, 2)
        country = st.selectbox(
            "Country",
            [
                "Turkey",
                "Georgia",
                "Vietnam",
                "Thailand",
                "South Korea",
                "India",
                "Philippines",
            ],
        )
        checkin = st.selectbox(
            "Check-in", ["Flexible", "After 1:00", "After 2:00", "After 3:00"]
        )

    # Boolean features
    st.subheader("Amenities & Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        has_wifi = st.checkbox("WiFi")
        has_kitchen = st.checkbox("Kitchen")
        has_parking = st.checkbox("Parking")

    with col2:
        has_pool = st.checkbox("Pool")
        has_gym = st.checkbox("Gym")
        has_balcony = st.checkbox("Balcony")

    with col3:
        has_mountain_view = st.checkbox("Mountain View")
        has_valley_view = st.checkbox("Valley View")

    # Make prediction
    if st.button("🔮 Predict Price"):
        try:
            # Create feature vector
            features = {
                "rating": rating,
                "reviews": reviews,
                "bathrooms": bathrooms,
                "beds": beds,
                "guests": guests,
                "bedrooms": bedrooms,
                "country": country,
                "checkin": checkin,
                "has_wifi": has_wifi,
                "has_kitchen": has_kitchen,
                "has_parking": has_parking,
                "has_pool": has_pool,
                "has_gym": has_gym,
                "has_balcony": has_balcony,
                "has_mountain_view": has_mountain_view,
                "has_valley_view": has_valley_view,
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([features])

            # Make predictions with all models
            predictions = {}
            for model_name in trainer.models.keys():
                try:
                    pred = trainer.predict(model_name, input_df)
                    if pred is not None:
                        predictions[model_name] = pred[0]
                except:
                    continue

            if predictions:
                st.subheader("📊 Price Predictions")

                # Display predictions
                for model_name, pred in predictions.items():
                    st.metric(f"{model_name}", f"${pred:,.2f}")

                # Average prediction
                avg_pred = np.mean(list(predictions.values()))
                st.metric("🎯 Average Prediction", f"${avg_pred:,.2f}")

                # Prediction range
                min_pred = min(predictions.values())
                max_pred = max(predictions.values())
                st.info(f"📈 Prediction Range: ${min_pred:,.2f} - ${max_pred:,.2f}")

            else:
                st.error(
                    "❌ No predictions could be made. Please check your input data."
                )

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")


def show_about():
    st.header("ℹ️ About")

    st.markdown(
        """
    ### Airbnb Price Prediction Project
    
    This application provides a comprehensive solution for analyzing Airbnb listing data and predicting property prices using machine learning.
    
    #### Features:
    - **Data Analysis**: Comprehensive exploration of Airbnb datasets
    - **Feature Engineering**: Automated creation of relevant features
    - **Model Training**: Multiple ML algorithms including Linear Regression, Random Forest, XGBoost, and LightGBM
    - **Price Prediction**: Interactive interface for making price predictions
    - **Performance Evaluation**: Detailed model comparison and evaluation
    
    #### Technology Stack:
    - **Backend**: Python, scikit-learn, XGBoost, LightGBM
    - **Frontend**: Streamlit
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Data Processing**: Pandas, NumPy
    
    #### Dataset Requirements:
    Your CSV should include columns like price, rating, reviews, bathrooms, beds, guests, bedrooms, country, and amenities.
    
    #### Usage:
    1. Upload your Airbnb dataset
    2. Explore the data using various analysis tools
    3. Train machine learning models
    4. Make price predictions for new listings
    
    For questions or support, please refer to the project documentation.
    """
    )


if __name__ == "__main__":
    main()




