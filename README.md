# Airbnb Price Prediction App

A Streamlit web application for analyzing Airbnb data and predicting property prices using machine learning.

## Getting Started

### 1. Navigate to Project Directory

```bash
cd "d:\projects\Airbnb analysis"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**

### App Preview

When you run the app, you'll see this interface:

![Airbnb Price Prediction App](<screenshot(187).png>)

**Note**: To add the actual screenshot:

1. Take a screenshot of your running app
2. Save it as `app_screenshot.png` in the project root directory
3. The image will automatically appear here

**Features you'll see:**

- **Home Page**: Welcome message and app overview
- **Data Upload**: Upload your Airbnb CSV dataset
- **Data Analysis**: Interactive data exploration tools
- **Model Training**: Train machine learning models
- **Price Prediction**: Input property details and get predictions
- **About**: Project information and documentation

## What the App Does

- **Data Analysis**: Upload and explore your Airbnb dataset
- **Model Training**: Train multiple ML algorithms (Linear Regression, Random Forest, XGBoost, LightGBM)
- **In progress - Price Prediction**: Input property details and get price predictions
- **In progress - Interactive Visualizations**: Charts, heatmaps, and dashboards

## Project Structure

```
├── app.py                 # Main Streamlit application
├── data/                  # Data storage
├── src/                   # Core modules
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Python dependencies
```

## Dataset Requirements

Your CSV should include these columns:

- `price` - Target variable (property price)
- `rating` - Property rating (1-5)
- `reviews` - Number of reviews
- `bathrooms`, `bedrooms`, `beds`, `guests`
- `country`, `address`
- Amenity features (WiFi, kitchen, parking, etc.)

## Troubleshooting

If the app doesn't start:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version (3.8+ recommended)
3. Verify you're in the correct directory

## Usage

1. **Upload Data**: Use the Data Upload page to upload your Airbnb CSV
2. **Explore Data**: Analyze your dataset with various visualization tools
3. **In progress - Train Models**: Click "Start Model Training" to build prediction models
4. **In progress - Make Predictions**: Input property details and get price estimates

---

**To start the app?** Just run `streamlit run app.py` in the project directory!
