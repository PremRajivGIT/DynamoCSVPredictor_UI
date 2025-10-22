import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error, mean_squared_log_error
import joblib
import os

# Check if custom CSV exists, otherwise use California dataset
csv_file = 'housing_data.csv'

if os.path.exists(csv_file):
    print(f"Loading custom dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Auto-detect target column (assumes last column is price/target)
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    # Handle non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric[df_numeric.columns[:-1]].values
    y = df_numeric[target_col].values
    
    feature_names = list(df_numeric.columns[:-1])
    target_name = target_col
    
    print(f"Target column: {target_name}")
    print(f"Features: {feature_names}")
    
else:
    print("No custom CSV found. Using California housing dataset...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target * 100000 * 83  # Convert to INR
    
    X = df[housing.feature_names].values
    y = df['Price'].values
    
    feature_names = list(housing.feature_names)
    target_name = 'Price'

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
explained_var = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
msle = mean_squared_log_error(np.abs(y_test), np.abs(y_pred))

# Save model, scaler, and metadata
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'feature_names': feature_names, 'target_name': target_name}, 'metadata.pkl')

print(f"\n{'='*60}")
print(f"Model Performance Metrics")
print(f"{'='*60}")
print(f"\nREGRESSION METRICS:")
print(f"R² Score (Coefficient of Determination): {r2:.6f}")
print(f"Adjusted R² Score: {1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1):.6f}")
print(f"Explained Variance Score: {explained_var:.6f}")
print(f"\nERROR METRICS:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"Mean Squared Log Error (MSLE): {msle:.6f}")
print(f"Max Error: {max_err:.2f}")
print(f"\nDATA STATISTICS:")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Features Used: {X.shape[1]}")
print(f"Target Range: {y.min():,.2f} - {y.max():,.2f}")
print(f"Mean Target: {y.mean():,.2f}")
print(f"Target Std Dev: {y.std():,.2f}")
print(f"{'='*60}\n")