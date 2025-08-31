"""
Utility functions for Airbnb price prediction.
Helper functions for data analysis, visualization, and common operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")


def load_and_validate_data(file_path):
    """Load data and perform basic validation."""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully! Shape: {df.shape}")

        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty!")

        if "price" not in df.columns and "Price" not in df.columns:
            print("⚠ Warning: Price column not found!")

        return df

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def quick_data_overview(df):
    """Provide a quick overview of the dataset."""
    print("=" * 60)
    print("QUICK DATA OVERVIEW")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()

        print(
            f"{i:2d}. {col:20s} | {dtype:12s} | Missing: {missing:4d} ({missing_pct:5.1f}%) | Unique: {unique:4d}"
        )

    print(f"\nMissing values total: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")


def analyze_price_distribution(df, price_col="price"):
    """Analyze the distribution of the price column."""
    if price_col not in df.columns:
        print(f"Price column '{price_col}' not found!")
        return

    # Convert to numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    print("=" * 60)
    print("PRICE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Basic statistics
    stats = df[price_col].describe()
    print("Basic Statistics:")
    print(stats)

    # Price ranges
    print(f"\nPrice Ranges:")
    print(f"Min: ${df[price_col].min():,.2f}")
    print(f"25th percentile: ${df[price_col].quantile(0.25):,.2f}")
    print(f"Median: ${df[price_col].median():,.2f}")
    print(f"75th percentile: ${df[price_col].quantile(0.75):,.2f}")
    print(f"Max: ${df[price_col].max():,.2f}")

    # Outlier detection
    Q1 = df[price_col].quantile(0.25)
    Q3 = df[price_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[price_col] < lower_bound) | (df[price_col] > upper_bound)]
    print(f"\nOutliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    print(f"Outlier range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

    return stats, outliers


def create_price_visualizations(df, price_col="price"):
    """Create comprehensive price visualizations."""
    if price_col not in df.columns:
        print(f"Price column '{price_col}' not found!")
        return

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Histogram
    axes[0, 0].hist(df[price_col].dropna(), bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Price Distribution")
    axes[0, 0].set_xlabel("Price")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Box plot
    axes[0, 1].boxplot(df[price_col].dropna())
    axes[0, 1].set_title("Price Box Plot")
    axes[0, 1].set_ylabel("Price")

    # 3. Log-transformed histogram
    log_prices = np.log1p(df[price_col].dropna())
    axes[0, 2].hist(log_prices, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("Log-Transformed Price Distribution")
    axes[0, 2].set_xlabel("Log(Price + 1)")
    axes[0, 2].set_ylabel("Frequency")

    # 4. Q-Q plot
    from scipy import stats

    stats.probplot(df[price_col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")

    # 5. Cumulative distribution
    sorted_prices = np.sort(df[price_col].dropna())
    cumulative = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
    axes[1, 1].plot(sorted_prices, cumulative)
    axes[1, 1].set_title("Cumulative Price Distribution")
    axes[1, 1].set_xlabel("Price")
    axes[1, 1].set_ylabel("Cumulative Probability")

    # 6. Price by country (if available)
    if "country" in df.columns:
        country_price = (
            df.groupby("country")[price_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        country_price.plot(kind="bar", ax=axes[1, 2])
        axes[1, 2].set_title("Average Price by Country (Top 10)")
        axes[1, 2].set_xlabel("Country")
        axes[1, 2].set_ylabel("Average Price")
        axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_categorical_features(df, categorical_cols=None):
    """Analyze categorical features and their relationship with price."""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not categorical_cols:
        print("No categorical columns found!")
        return

    print("=" * 60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 60)

    for col in categorical_cols:
        if col in df.columns:
            print(f"\n--- {col.upper()} ---")

            # Basic info
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()

            print(f"Unique values: {unique_count}")
            print(f"Missing values: {missing_count}")

            # Value counts
            if unique_count <= 20:
                value_counts = df[col].value_counts()
                print("Top values:")
                print(value_counts.head(10))

                # Price analysis by category
                if "price" in df.columns:
                    price_by_category = df.groupby(col)["price"].agg(
                        ["mean", "median", "count"]
                    )
                    print(f"\nPrice by {col}:")
                    print(
                        price_by_category.sort_values("mean", ascending=False).head(5)
                    )
            else:
                print(f"Too many unique values ({unique_count}) to display all")
                print("Top 10 values:")
                print(df[col].value_counts().head(10))


def create_correlation_heatmap(df, numerical_cols=None):
    """Create correlation heatmap for numerical features."""
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        print("Need at least 2 numerical columns for correlation analysis!")
        return

    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.show()

    # Show top correlations with price
    if "price" in numerical_cols:
        price_correlations = correlation_matrix["price"].sort_values(ascending=False)
        print("\nTop correlations with price:")
        print(price_correlations.head(10))


def create_interactive_dashboard(df, price_col="price"):
    """Create an interactive dashboard using Plotly."""
    if price_col not in df.columns:
        print(f"Price column '{price_col}' not found!")
        return

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Price Distribution",
            "Price by Rating",
            "Price by Number of Guests",
            "Price by Country",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # 1. Price distribution
    fig.add_trace(
        go.Histogram(x=df[price_col].dropna(), name="Price Distribution", nbinsx=50),
        row=1,
        col=1,
    )

    # 2. Price by rating
    if "rating" in df.columns:
        rating_price = df.groupby("rating")[price_col].mean()
        fig.add_trace(
            go.Scatter(
                x=rating_price.index,
                y=rating_price.values,
                mode="lines+markers",
                name="Price by Rating",
            ),
            row=1,
            col=2,
        )

    # 3. Price by guests
    if "guests" in df.columns:
        guests_price = df.groupby("guests")[price_col].mean()
        fig.add_trace(
            go.Bar(x=guests_price.index, y=guests_price.values, name="Price by Guests"),
            row=2,
            col=1,
        )

    # 4. Price by country
    if "country" in df.columns:
        country_price = (
            df.groupby("country")[price_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig.add_trace(
            go.Bar(
                x=country_price.index, y=country_price.values, name="Price by Country"
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title="Airbnb Price Analysis Dashboard", height=800, showlegend=True
    )

    fig.show()


def save_analysis_results(results, filename):
    """Save analysis results to a file."""
    try:
        with open(filename, "w") as f:
            f.write(str(results))
        print(f"✓ Analysis results saved to {filename}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


def main():
    """Main function for testing utilities."""
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("- load_and_validate_data()")
    print("- quick_data_overview()")
    print("- analyze_price_distribution()")
    print("- create_price_visualizations()")
    print("- analyze_categorical_features()")
    print("- create_correlation_heatmap()")
    print("- create_interactive_dashboard()")
    print("- save_analysis_results()")


if __name__ == "__main__":
    main()

