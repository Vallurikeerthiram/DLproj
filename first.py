"""
Simple Water Intake Prediction Script
This is a streamlined version for quick execution
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
print("Loading data...")
try:
    df = pd.read_excel('DLdatas (1).xlsx')
except FileNotFoundError:
    try:
        df = pd.read_excel('/workspaces/DLproj/DLdatas (1).xlsx')
    except FileNotFoundError:
        print("Error: Could not find data file!")
        print("Please ensure 'DLdatas (1).xlsx' is in the current directory")
        exit(1)

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Prepare data
df = df.dropna()

# Identify target column
target_candidates = [col for col in df.columns if 'water' in col.lower() or 'intake' in col.lower()]
target_col = target_candidates[0] if target_candidates else df.columns[-1]

print(f"Target column: {target_col}")

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle non-numeric columns
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_datetime(X[col]).astype(int) / 10**9
        except:
            X = pd.get_dummies(X, columns=[col], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Build model
print("\nBuilding model...")
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE:      {mae:.4f}")
print(f"MSE:      {mse:.4f}")
print(f"RMSE:     {np.sqrt(mse):.4f}")
print("=" * 50)

# Save model
model.save('best_model.h5')
print("\nModel saved as 'best_model.h5'")
print("✓ Done!")
