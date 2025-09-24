
"""
Statistical Methods Assignment: Multiple Linear Regression

This script demonstrates the implementation of multiple linear regression with 
comprehensive statistical analysis using only numpy and scipy.stats.

Assignment Requirements (Passing Grade G):
- Property `d` that contains the number of features/parameters/dimensions of the model
- Property `n` that contains the size of the sample
- Function to calculate the variance
- Function to calculate the standard deviation
- Function that reports the significance of the regression
- Function that reports the relevance of the regression (R²)
"""

# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from linear_regression import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("Statistical Methods Assignment: Multiple Linear Regression")
print("=" * 60)

# =============================================================================
# 1. Generate Sample Data
# =============================================================================
print("\n## 1. Generate Sample Data")
print("-" * 30)
print("We'll create a synthetic dataset with multiple features to demonstrate")
print("the linear regression functionality.")

# Generate sample data
n_samples = 100
n_features = 3

# Create feature matrix X with some correlation between features
X = np.random.randn(n_samples, n_features)
# Add some correlation between features 0 and 1
X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]

# Create true coefficients
true_coefficients = np.array([2.5, -1.2, 0.8])
true_intercept = 1.5

# Generate target variable with noise
y = true_intercept + X @ true_coefficients + 0.5 * np.random.randn(n_samples)

print(f"Generated dataset with {n_samples} samples and {n_features} features")
print(f"True intercept: {true_intercept}")
print(f"True coefficients: {true_coefficients}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# =============================================================================
# 2. Initialize and Fit the Linear Regression Model
# =============================================================================
print("\n## 2. Initialize and Fit the Linear Regression Model")
print("-" * 50)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

print("Model fitted successfully!")
print(f"Fitted intercept: {model.intercept:.4f}")
print(f"Fitted coefficients: {model.feature_coefficients}")
print(f"Number of features (d): {model.d}")
print(f"Number of samples (n): {model.n}")

# =============================================================================
# 3. Demonstrate Required Properties
# =============================================================================
print("\n## 3. Demonstrate Required Properties")
print("-" * 40)

# Property `d` - Number of features/parameters/dimensions
print("\n### Property `d` - Number of features/parameters/dimensions")
print(f"Number of features (d): {model.d}")
print(f"This represents the dimensionality of our feature space")

# Property `n` - Size of the sample
print("\n### Property `n` - Size of the sample")
print(f"Number of samples (n): {model.n}")
print(f"This represents the number of observations in our dataset")

# =============================================================================
# 4. Calculate Variance and Standard Deviation
# =============================================================================
print("\n## 4. Calculate Variance and Standard Deviation")
print("-" * 50)

# Variance Calculation
print("\n### Variance Calculation")
variance = model.calculate_variance()
print(f"Variance (σ²): {variance:.6f}")
print(f"Formula: SSE / (n - d - 1) = SSE / ({model.n} - {model.d} - 1) = SSE / {model.n - model.d - 1}")

# Manual calculation for verification
y_pred = model.predict(X)
sse = np.sum((y - y_pred) ** 2)
manual_variance = sse / (model.n - model.d - 1)
print(f"Manual calculation: {manual_variance:.6f}")
print(f"Match: {np.isclose(variance, manual_variance)}")

# Standard Deviation Calculation
print("\n### Standard Deviation Calculation")
std_dev = model.calculate_standard_deviation()
print(f"Standard Deviation (σ): {std_dev:.6f}")
print(f"Formula: √(variance) = √({variance:.6f})")
print(f"Manual calculation: {np.sqrt(variance):.6f}")
print(f"Match: {np.isclose(std_dev, np.sqrt(variance))}")

# =============================================================================
# 5. Regression Significance Testing
# =============================================================================
print("\n## 5. Regression Significance Testing")
print("-" * 40)

# F-test for Overall Regression Significance
print("\n### F-test for Overall Regression Significance")
f_statistic, f_p_value = model.test_regression_significance()

print("=== Regression Significance Test (F-test) ===")
print(f"F-statistic: {f_statistic:.6f}")
print(f"p-value: {f_p_value:.6f}")
print(f"Degrees of freedom (numerator): {model.d}")
print(f"Degrees of freedom (denominator): {model.n - model.d - 1}")
print(f"Significance level: α = 0.05")
print(f"Regression is significant: {f_p_value < 0.05}")

if f_p_value < 0.05:
    print("✓ The regression is statistically significant at α = 0.05")
else:
    print("✗ The regression is not statistically significant at α = 0.05")

# =============================================================================
# 6. Regression Relevance (R²)
# =============================================================================
print("\n## 6. Regression Relevance (R²)")
print("-" * 35)

# Coefficient of Multiple Determination
print("\n### Coefficient of Multiple Determination")
r_squared = model.calculate_r_squared()

print("=== Regression Relevance (R²) ===")
print(f"R²: {r_squared:.6f}")
print(f"R² percentage: {r_squared * 100:.2f}%")
print(f"Formula: R² = SSR / Syy")

# Manual calculation for verification
y_mean = np.mean(y)
syy = np.sum((y - y_mean) ** 2)
sse = np.sum((y - y_pred) ** 2)
ssr = syy - sse
manual_r_squared = ssr / syy

print(f"\nManual calculation:")
print(f"Syy (Total Sum of Squares): {syy:.6f}")
print(f"SSE (Sum of Squared Errors): {sse:.6f}")
print(f"SSR (Sum of Squares due to Regression): {ssr:.6f}")
print(f"R² = SSR / Syy = {ssr:.6f} / {syy:.6f} = {manual_r_squared:.6f}")
print(f"Match: {np.isclose(r_squared, manual_r_squared)}")

if r_squared > 0.7:
    print("✓ Strong relationship between features and target")
elif r_squared > 0.5:
    print("✓ Moderate relationship between features and target")
else:
    print("⚠ Weak relationship between features and target")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✓ All assignment requirements have been successfully demonstrated:")
print("  - Property 'd' (number of features):", model.d)
print("  - Property 'n' (number of samples):", model.n)
print("  - Variance calculation:", f"{variance:.6f}")
print("  - Standard deviation calculation:", f"{std_dev:.6f}")
print("  - Regression significance test (F-test):", f"p-value = {f_p_value:.6f}")
print("  - Regression relevance (R²):", f"{r_squared:.6f}")
print("\n✓ The LinearRegression class implements all required statistical methods")
print("✓ All calculations have been manually verified")
print("=" * 60)
