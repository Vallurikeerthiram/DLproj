"""
Automated Water Intake Prediction Model Training Script
This script fully automates the process of loading data, training a deep learning model,
and evaluating it with comprehensive metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class WaterIntakePredictor:
    """Automated water intake prediction model"""
    
    def __init__(self, data_path='DLdatas (1).xlsx'):
        """
        Initialize the predictor
        
        Args:
            data_path: Path to the Excel data file
        """
        self.data_path = data_path
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare data from Excel file"""
        print("=" * 70)
        print("STEP 1: Loading Data")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            print(f"ERROR: Data file '{self.data_path}' not found!")
            print("Please ensure the Excel file is in the current directory.")
            # Try alternative paths
            alt_paths = [
                '/workspaces/DLproj/DLdatas (1).xlsx',
                'DLdatas.xlsx',
                'data.xlsx'
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.data_path = alt_path
                    print(f"Found data at: {alt_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find data file. Tried: {self.data_path}, {alt_paths}")
        
        # Load the data
        df = pd.read_excel(self.data_path)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Display basic statistics
        print("\nData Overview:")
        print(df.head())
        print("\nData Info:")
        print(df.info())
        print("\nStatistical Summary:")
        print(df.describe())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess and prepare data for training"""
        print("\n" + "=" * 70)
        print("STEP 2: Preprocessing Data")
        print("=" * 70)
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        df = df.dropna()
        print(f"✓ Removed rows with missing values. New shape: {df.shape}")
        
        # Identify feature and target columns
        # Assuming the last column or a column with 'water' or 'intake' in name is the target
        target_candidates = [col for col in df.columns if 'water' in col.lower() or 'intake' in col.lower()]
        
        if target_candidates:
            target_col = target_candidates[0]
        else:
            # Use the last column as target
            target_col = df.columns[-1]
        
        print(f"✓ Target column identified: '{target_col}'")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Convert non-numeric columns to numeric if needed
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_datetime(X[col]).astype(int) / 10**9  # Convert to timestamp
                except:
                    # Try one-hot encoding for categorical
                    X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"  Feature columns: {list(X.columns)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
        # Scale the data
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_test = self.scaler_X.transform(self.X_test)
        
        self.y_train = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
        self.y_test = self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel()
        
        print("✓ Data normalized using StandardScaler")
        
        return X.shape[1]  # Return number of features
    
    def build_model(self, input_dim):
        """Build a deep neural network model optimized for high R² score"""
        print("\n" + "=" * 70)
        print("STEP 3: Building Deep Learning Model")
        print("=" * 70)
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(input_dim,),
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers with decreasing neurons
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            
            # Output layer
            layers.Dense(1)
        ])
        
        # Compile with Adam optimizer and appropriate learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("✓ Model architecture created:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=200, batch_size=32):
        """Train the model with callbacks for optimal performance"""
        print("\n" + "=" * 70)
        print("STEP 4: Training Model")
        print("=" * 70)
        
        # Define callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train the model
        print(f"Training with {epochs} epochs and batch size {batch_size}...")
        print("This may take a few minutes...")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
    def evaluate_model(self):
        """Evaluate the model and display comprehensive metrics"""
        print("\n" + "=" * 70)
        print("STEP 5: Model Evaluation")
        print("=" * 70)
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(self.X_train, verbose=0)
        y_test_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform to original scale
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled).ravel()
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled).ravel()
        y_train_actual = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1)).ravel()
        y_test_actual = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        
        # Calculate metrics for training set
        train_r2 = r2_score(y_train_actual, y_train_pred)
        train_mae = mean_absolute_error(y_train_actual, y_train_pred)
        train_mse = mean_squared_error(y_train_actual, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        
        # Calculate metrics for test set
        test_r2 = r2_score(y_test_actual, y_test_pred)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred)
        test_mse = mean_squared_error(y_test_actual, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        
        # Display results
        print("\n" + "=" * 70)
        print("TRAINING SET METRICS")
        print("=" * 70)
        print(f"R² Score:  {train_r2:.4f} ({train_r2*100:.2f}%)")
        print(f"MAE:       {train_mae:.4f}")
        print(f"MSE:       {train_mse:.4f}")
        print(f"RMSE:      {train_rmse:.4f}")
        
        print("\n" + "=" * 70)
        print("TEST SET METRICS")
        print("=" * 70)
        print(f"R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)")
        print(f"MAE:       {test_mae:.4f}")
        print(f"MSE:       {test_mse:.4f}")
        print(f"RMSE:      {test_rmse:.4f}")
        
        # Performance assessment
        print("\n" + "=" * 70)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 70)
        
        if test_r2 >= 0.6:
            print("🎉 EXCELLENT! R² Score is above 60% - Outstanding performance!")
        elif test_r2 >= 0.5:
            print("✓ GOOD! R² Score is above 50% - Target achieved!")
        elif test_r2 >= 0.4:
            print("→ MODERATE. R² Score is above 40% - Decent performance.")
        else:
            print("⚠ NEEDS IMPROVEMENT. R² Score is below 40%.")
        
        # Overfitting check
        if train_r2 - test_r2 > 0.15:
            print("⚠ Warning: Model shows signs of overfitting (train R² >> test R²)")
        else:
            print("✓ Model generalizes well (no significant overfitting)")
        
        return {
            'train': {'r2': train_r2, 'mae': train_mae, 'mse': train_mse, 'rmse': train_rmse},
            'test': {'r2': test_r2, 'mae': test_mae, 'mse': test_mse, 'rmse': test_rmse}
        }
    
    def plot_results(self, metrics):
        """Create visualization plots"""
        print("\n" + "=" * 70)
        print("STEP 6: Creating Visualizations")
        print("=" * 70)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Water Intake Prediction Model - Comprehensive Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Training History - Loss
        ax1 = axes[0, 0]
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training History - MAE
        ax2 = axes[0, 1]
        ax2.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Mean Absolute Error Over Epochs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Metrics Comparison
        ax3 = axes[1, 0]
        metrics_data = {
            'R² Score': [metrics['train']['r2'], metrics['test']['r2']],
            'MAE': [metrics['train']['mae'], metrics['test']['mae']],
            'RMSE': [metrics['train']['rmse'], metrics['test']['rmse']]
        }
        x = np.arange(len(metrics_data))
        width = 0.35
        train_values = [metrics['train']['r2'], metrics['train']['mae'], metrics['train']['rmse']]
        test_values = [metrics['test']['r2'], metrics['test']['mae'], metrics['test']['rmse']]
        
        ax3.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
        ax3.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['R² Score', 'MAE', 'RMSE'])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Predictions vs Actual (Test Set)
        ax4 = axes[1, 1]
        y_test_pred_scaled = self.model.predict(self.X_test, verbose=0)
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled).ravel()
        y_test_actual = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        
        ax4.scatter(y_test_actual, y_test_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_test_actual.min(), y_test_pred.min())
        max_val = max(y_test_actual.max(), y_test_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax4.set_xlabel('Actual Water Intake', fontsize=12)
        ax4.set_ylabel('Predicted Water Intake', fontsize=12)
        ax4.set_title(f'Predictions vs Actual (Test Set)\nR² = {metrics["test"]["r2"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        print("✓ Results plot saved as 'training_results.png'")
        
        try:
            plt.show()
        except:
            print("  (Display not available in this environment)")
    
    def save_model(self, filepath='best_model.h5'):
        """Save the trained model"""
        print("\n" + "=" * 70)
        print("STEP 7: Saving Model")
        print("=" * 70)
        
        self.model.save(filepath)
        print(f"✓ Model saved successfully to '{filepath}'")
        print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    def run_complete_pipeline(self):
        """Execute the complete automated pipeline"""
        print("\n" + "=" * 70)
        print("WATER INTAKE PREDICTION - AUTOMATED TRAINING PIPELINE")
        print("=" * 70)
        print("This script will automatically:")
        print("  1. Load and explore the data")
        print("  2. Preprocess and prepare features")
        print("  3. Build a deep learning model")
        print("  4. Train with optimal hyperparameters")
        print("  5. Evaluate with comprehensive metrics")
        print("  6. Generate visualizations")
        print("  7. Save the best model")
        print("=" * 70)
        
        try:
            # Execute pipeline steps
            df = self.load_data()
            input_dim = self.preprocess_data(df)
            self.build_model(input_dim)
            self.train_model(epochs=200, batch_size=32)
            metrics = self.evaluate_model()
            self.plot_results(metrics)
            self.save_model('best_model.h5')
            
            # Final summary
            print("\n" + "=" * 70)
            print("AUTOMATION COMPLETE!")
            print("=" * 70)
            print(f"✓ Model trained and evaluated successfully")
            print(f"✓ Test R² Score: {metrics['test']['r2']:.4f} ({metrics['test']['r2']*100:.2f}%)")
            print(f"✓ Test MAE: {metrics['test']['mae']:.4f}")
            print(f"✓ Test MSE: {metrics['test']['mse']:.4f}")
            print(f"✓ Model saved to 'best_model.h5'")
            print(f"✓ Visualizations saved to 'training_results.png'")
            print("=" * 70)
            
            return metrics
            
        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR OCCURRED!")
            print("=" * 70)
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("Starting Automated Water Intake Prediction Pipeline")
    print("=" * 70)
    
    # Create predictor instance
    predictor = WaterIntakePredictor()
    
    # Run complete pipeline
    results = predictor.run_complete_pipeline()
    
    if results:
        print("\n🎉 SUCCESS! All tasks completed successfully!")
    else:
        print("\n⚠ Pipeline completed with errors. Please check the output above.")


if __name__ == "__main__":
    main()
