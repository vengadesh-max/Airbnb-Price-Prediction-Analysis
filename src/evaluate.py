"""
Model evaluation module for Airbnb price prediction.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, max_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """Evaluate a single model with multiple metrics."""
        print(f"Evaluating {model_name}...")

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)

        # Calculate percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Store results
        self.evaluation_results[model_name] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Explained Variance": explained_var,
            "Max Error": max_err,
            "MAPE": mape,
        }

        # Print results
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return self.evaluation_results[model_name]

    def compare_models(self, results_dict):
        """Compare multiple models and display results."""
        print("Comparing models...")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results_dict).T

        # Display results
        print("\nModel Comparison:")
        display(comparison_df.round(4))

        # Find best model for each metric
        best_models = {}
        for metric in ["R2", "RMSE", "MAE", "MAPE"]:
            if metric == "R2":
                best_model = comparison_df[metric].idxmax()
                best_value = comparison_df[metric].max()
            else:
                best_model = comparison_df[metric].idxmin()
                best_value = comparison_df[metric].min()

            best_models[metric] = (best_model, best_value)
            print(f"Best {metric}: {best_model} ({best_value:.4f})")

        return comparison_df, best_models

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name="Model"):
        """Plot predicted vs actual values."""
        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"{model_name}: Actual vs Predicted")

        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.title(f"{model_name}: Residuals Plot")

        plt.tight_layout()
        plt.show()

        # Residuals statistics
        print(f"\nResiduals Statistics for {model_name}:")
        print(f"Mean: {residuals.mean():.2f}")
        print(f"Std: {residuals.std():.2f}")
        print(f"Min: {residuals.min():.2f}")
        print(f"Max: {residuals.max():.2f}")

    def plot_error_distribution(self, y_true, y_pred, model_name="Model"):
        """Plot error distribution."""
        errors = y_true - y_pred

        plt.figure(figsize=(12, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"{model_name}: Error Distribution")

        # Q-Q plot
        plt.subplot(1, 2, 2)
        from scipy import stats

        stats.probplot(errors, dist="norm", plot=plt)
        plt.title(f"{model_name}: Q-Q Plot (Normal Distribution)")

        plt.tight_layout()
        plt.show()

    def create_interactive_evaluation(self, y_true, y_pred, model_name="Model"):
        """Create interactive evaluation plots using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Actual vs Predicted",
                "Residuals Plot",
                "Error Distribution",
                "Residuals vs Predicted",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(color="blue", opacity=0.6),
            ),
            row=1,
            col=1,
        )

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="green", opacity=0.6),
            ),
            row=1,
            col=2,
        )

        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode="lines",
                name="Zero Line",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=2,
        )

        # Error distribution
        fig.add_trace(
            go.Histogram(x=errors, name="Error Distribution", nbinsx=30), row=2, col=1
        )

        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals vs Pred",
                marker=dict(color="orange", opacity=0.6),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"{model_name} - Comprehensive Evaluation",
            height=800,
            showlegend=True,
        )

        fig.show()

    def generate_evaluation_report(self, output_file=None):
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            print("No evaluation results available!")
            return

        report = "=" * 60 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"

        # Summary table
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"

        df = pd.DataFrame(self.evaluation_results).T
        report += df.to_string()

        # Best models
        report += "\n\nBEST PERFORMING MODELS:\n"
        report += "-" * 30 + "\n"

        for metric in ["R2", "RMSE", "MAE"]:
            if metric == "R2":
                best_model = df[metric].idxmax()
                best_value = df[metric].max()
            else:
                best_model = df[metric].idxmin()
                best_value = df[metric].min()

            report += f"{metric}: {best_model} ({best_value:.4f})\n"

        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"

        best_r2_model = df["R2"].idxmax()
        best_r2_score = df["R2"].max()

        if best_r2_score > 0.8:
            report += f"✓ {best_r2_model} shows excellent performance (R² = {best_r2_score:.4f})\n"
        elif best_r2_score > 0.6:
            report += (
                f"✓ {best_r2_model} shows good performance (R² = {best_r2_score:.4f})\n"
            )
        else:
            report += (
                f"⚠ All models show poor performance. Consider feature engineering.\n"
            )

        # Save report
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Evaluation report saved to {output_file}")

        print(report)
        return report


def main():
    """Main function for testing the evaluator."""
    evaluator = ModelEvaluator()
    print("ModelEvaluator class created successfully!")
    print("Use this class to evaluate your ML models.")


if __name__ == "__main__":
    main()

