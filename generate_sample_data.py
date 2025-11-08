"""
Generate sample water intake data for testing
This creates a synthetic dataset similar to what might be expected
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 500

# Create date range
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_samples)]

# Generate features
data = {
    'Date': dates,
    'Temperature': np.random.uniform(15, 35, n_samples),  # Temperature in Celsius
    'Humidity': np.random.uniform(30, 90, n_samples),     # Humidity percentage
    'Activity_Level': np.random.uniform(0, 10, n_samples), # Activity level (0-10)
    'Body_Weight': np.random.uniform(50, 100, n_samples),  # Weight in kg
    'Age': np.random.randint(18, 70, n_samples),           # Age in years
    'Exercise_Duration': np.random.uniform(0, 120, n_samples), # Minutes
}

# Generate target variable (water intake) with some correlation to features
# Water intake increases with temperature, activity, exercise, and body weight
water_intake = (
    1000 +  # Base intake
    data['Temperature'] * 30 +  # More water in hot weather
    data['Activity_Level'] * 50 +  # More water with activity
    data['Exercise_Duration'] * 3 +  # More water with exercise
    data['Body_Weight'] * 10 +  # More water for higher body weight
    data['Humidity'] * -2 +  # Slightly less in high humidity
    np.random.normal(0, 200, n_samples)  # Random variation
)

# Ensure water intake is positive and reasonable (500ml to 5000ml)
water_intake = np.clip(water_intake, 500, 5000)

data['Water_Intake_ml'] = water_intake

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = 'DLdatas (1).xlsx'
df.to_excel(output_file, index=False)

print(f"✓ Sample data generated successfully!")
print(f"  File: {output_file}")
print(f"  Samples: {n_samples}")
print(f"  Columns: {list(df.columns)}")
print(f"\nData preview:")
print(df.head(10))
print(f"\nStatistics:")
print(df.describe())
print(f"\nNote: This is synthetic data for demonstration purposes.")
print(f"Replace with your actual data file to train on real data.")
