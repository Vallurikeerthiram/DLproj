# DLproj - Water Intake Prediction Using Deep Learning

This project uses deep learning to predict water intake based on historical data, achieving high R² scores with comprehensive evaluation metrics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model

#### Option 1: Full Automated Pipeline (Recommended)
Run the comprehensive training script with detailed logging and visualizations:
```bash
python train_model.py
```

This script will:
- Automatically load and explore the data
- Preprocess and prepare features
- Build an optimized deep learning model
- Train with early stopping and learning rate reduction
- Evaluate with R², MAE, MSE, and RMSE metrics
- Generate visualization plots
- Save the best model

#### Option 2: Simple Training Script
For a quick training run:
```bash
python first.py
```

## 📊 Features

- **Automated Data Loading**: Handles Excel files with automatic column detection
- **Smart Preprocessing**: Automatic feature engineering and scaling
- **Deep Neural Network**: Multi-layer architecture with dropout and batch normalization
- **Comprehensive Metrics**: R², MAE, MSE, RMSE for thorough evaluation
- **Visualization**: Training history plots and prediction comparisons
- **Model Persistence**: Saves the best model for future use

## 📈 Model Performance

The model is designed to achieve:
- **Target R² Score**: > 0.5 (50%) - 0.6 (60%)
- **Low MAE**: Minimized mean absolute error
- **Optimized MSE**: Mean squared error optimization
- **Good Generalization**: Minimal overfitting between train and test sets

## 🗂️ Project Structure

```
DLproj/
├── train_model.py          # Comprehensive automated training pipeline
├── first.py                # Simple training script
├── requirements.txt        # Python dependencies
├── best_model.h5          # Saved trained model
├── training_results.png   # Visualization outputs (generated)
└── README.md              # This file
```

## 📦 Dependencies

- TensorFlow 2.13+
- pandas 2.0+
- scikit-learn 1.3+
- matplotlib 3.7+
- numpy 1.24+
- openpyxl 3.1+
- seaborn 0.12+

## 🎯 Data Requirements

The model expects an Excel file (`DLdatas (1).xlsx`) with:
- Historical water intake data
- Multiple feature columns
- A target column containing water intake values (will be auto-detected)

## 🔧 Customization

You can modify the model architecture, hyperparameters, or training settings by editing:
- `train_model.py` for the full pipeline
- `first.py` for quick experiments

## 📝 Output Files

After training:
- `best_model.h5` - Trained model (can be loaded for predictions)
- `training_results.png` - Comprehensive visualization plots

## 🎓 Model Architecture

The deep learning model uses:
- Input layer with 256 neurons
- Multiple hidden layers (128, 64, 32, 16 neurons)
- ReLU activation functions
- Batch normalization for stability
- Dropout layers for regularization
- L2 regularization to prevent overfitting
- Adam optimizer with learning rate scheduling

## 💡 Tips for Best Results

1. Ensure your data is clean and properly formatted
2. The model automatically handles missing values
3. Feature scaling is applied automatically
4. Early stopping prevents overfitting
5. The best model weights are automatically restored

## 🤝 Contributing

This is an automated deep learning project. The scripts handle all preprocessing, training, and evaluation automatically.

## 📄 License

This project is open source and available for educational purposes.