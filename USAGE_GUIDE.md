# Water Intake Prediction - Complete Usage Guide

## 📋 Overview

This project provides a fully automated deep learning solution for predicting water intake based on historical data. The system handles everything from data loading to model evaluation automatically.

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script (Linux/Mac):
```bash
bash setup.sh
```

### Step 2: Prepare Your Data

Place your Excel file named `DLdatas (1).xlsx` in the project directory. The file should contain:
- Historical data with features (temperature, humidity, activity level, etc.)
- A target column with water intake values (will be auto-detected)

**Don't have data?** Generate sample data for testing:
```bash
python generate_sample_data.py
```

### Step 3: Train the Model

**Option A - Full Pipeline (Recommended)**
```bash
python train_model.py
```

**Option B - Quick Training**
```bash
python first.py
```

## 📊 What Happens During Training

The automated pipeline performs these steps:

1. **Data Loading**: Automatically loads Excel file and displays data overview
2. **Preprocessing**: 
   - Handles missing values
   - Auto-detects target column
   - Converts datetime columns
   - Scales features using StandardScaler
3. **Model Building**: Creates deep neural network with:
   - 5 hidden layers (256→128→64→32→16 neurons)
   - Dropout for regularization
   - Batch normalization for stability
4. **Training**: 
   - Up to 200 epochs with early stopping
   - Learning rate reduction on plateau
   - Validation split for monitoring
5. **Evaluation**: Calculates R², MAE, MSE, RMSE metrics
6. **Visualization**: Generates comprehensive plots
7. **Model Saving**: Saves best model as `best_model.h5`

## 📈 Understanding the Results

### Performance Metrics

- **R² Score**: Measures how well the model explains the variance
  - Above 60%: Excellent
  - 50-60%: Good (Target achieved!)
  - 40-50%: Moderate
  - Below 40%: Needs improvement

- **MAE (Mean Absolute Error)**: Average prediction error
  - Lower is better
  - Same units as target variable

- **MSE (Mean Squared Error)**: Squared average error
  - Lower is better
  - Penalizes large errors more

- **RMSE (Root Mean Squared Error)**: Square root of MSE
  - Same units as target variable
  - More interpretable than MSE

### Output Files

After training, you'll find:

1. **best_model.h5** (~0.6-1.5 MB)
   - The trained deep learning model
   - Can be loaded for making predictions

2. **training_results.png**
   - 4-panel visualization showing:
     - Training/validation loss over epochs
     - MAE over epochs
     - Metrics comparison (train vs test)
     - Predictions vs actual values scatter plot

## 🔧 Customization Options

### Modifying Hyperparameters

Edit `train_model.py` and change:

```python
# Training parameters
self.train_model(epochs=200, batch_size=32)

# Model architecture (in build_model method)
layers.Dense(256, activation='relu')  # Change neuron count
layers.Dropout(0.3)  # Change dropout rate
```

### Using Different Data Files

Edit the script and change:
```python
predictor = WaterIntakePredictor(data_path='your_file.xlsx')
```

## 🎯 Achieving Higher R² Scores

If you're not reaching your target R² score, try:

1. **More Data**: Collect more samples (aim for 500+ samples)
2. **Feature Engineering**: Add derived features
3. **Hyperparameter Tuning**: Adjust:
   - Learning rate
   - Number of neurons
   - Dropout rates
   - Batch size
4. **Longer Training**: Increase epochs (but watch for overfitting)
5. **Data Quality**: Ensure data is clean and accurate

## 📝 Example Session

```bash
$ python train_model.py

======================================================================
WATER INTAKE PREDICTION - AUTOMATED TRAINING PIPELINE
======================================================================

STEP 1: Loading Data
======================================================================
✓ Data loaded successfully!
  Shape: (500, 8)
  
STEP 2: Preprocessing Data
======================================================================
✓ Target column identified: 'Water_Intake_ml'
✓ Train set: 400 samples
✓ Test set: 100 samples

STEP 3: Building Deep Learning Model
======================================================================
✓ Model architecture created

STEP 4: Training Model
======================================================================
Training with 200 epochs...
Epoch 1/200 ... loss: 0.8494 - val_loss: 0.9994
[Training continues...]

STEP 5: Model Evaluation
======================================================================
TEST SET METRICS
======================================================================
R² Score:  0.5181 (51.81%)
MAE:       190.1727
MSE:       53263.2092

✓ GOOD! R² Score is above 50% - Target achieved!

======================================================================
AUTOMATION COMPLETE!
======================================================================
✓ Model saved to 'best_model.h5'
✓ Visualizations saved to 'training_results.png'
```

## 🔄 Using the Trained Model

To load and use the model for predictions:

```python
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
model = tf.keras.models.load_model('best_model.h5')

# Prepare new data (same preprocessing as training)
new_data = pd.DataFrame({
    'Temperature': [25.0],
    'Humidity': [60.0],
    'Activity_Level': [5.0],
    # ... other features
})

# Scale the features (use same scaler as training)
# Note: In production, save the scaler along with the model
scaled_data = scaler.transform(new_data)

# Make prediction
prediction = model.predict(scaled_data)
print(f"Predicted water intake: {prediction[0][0]}")
```

## 🐛 Troubleshooting

### Error: "No module named 'tensorflow'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Error: "Could not find data file"
**Solution**: Ensure `DLdatas (1).xlsx` is in the current directory, or generate sample data:
```bash
python generate_sample_data.py
```

### Low R² Score
**Solutions**:
- Check data quality (no duplicates, correct values)
- Try collecting more data samples
- Adjust hyperparameters
- Ensure features are relevant to water intake

### Model overfitting (Train R² >> Test R²)
**Solutions**:
- Increase dropout rates
- Add more regularization
- Reduce model complexity
- Collect more training data

## 💡 Tips for Best Results

1. **Data Quality**: Clean, accurate data is crucial
2. **Feature Selection**: Use relevant features (temperature, activity, etc.)
3. **Sufficient Samples**: At least 200-500 samples recommended
4. **Run Multiple Times**: Neural networks have randomness; train several times
5. **Monitor Training**: Watch for overfitting in the plots

## 📧 Support

If you encounter issues:
1. Check this guide first
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify your data format matches expectations

## 🎓 Understanding the Code

### Simple Version (`first.py`)
- Streamlined, easy to understand
- Quick training (~2-5 minutes)
- Basic output

### Comprehensive Version (`train_model.py`)
- Detailed logging at every step
- Advanced callbacks (early stopping, LR reduction)
- Comprehensive visualizations
- Better organized with class structure

Choose based on your needs:
- **Learning/Experimenting**: Use `first.py`
- **Production/Best Results**: Use `train_model.py`

## 📚 Additional Resources

- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Deep Learning Best Practices: Various online resources

---

**Happy Predicting! 🎉**
