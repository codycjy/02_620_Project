# Cell Prediction Package

A comprehensive machine learning solution for predicting cognitive status from cell expression data.

## Overview

This package provides a complete pipeline for processing cell expression data, training multiple machine learning models (Linear Regression, Random Forest, and optionally XGBoost), selecting the best performing model, and making predictions on new data. The package includes functionality for data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.


## Features

- **Comprehensive Feature Engineering**: Combines mean expressions across all clusters for each donor, creating a standardized feature set
- **Multiple Model Support**: Train and compare Linear Regression, Random Forest, and optionally XGBoost models
- **Automated Model Selection**: Select the best performing model based on validation metrics
- **Hyperparameter Tuning**: Grid search for optimal model parameters
- **Cross-Validation**: Evaluate model performance using k-fold cross-validation
- **Comprehensive Visualization**: Generate visualizations for model performance, feature importance, and data distributions
- **Flexible Data Processing**: Process cell expression data from various formats
- **Command-Line Interface**: Easy-to-use CLI for running the full pipeline

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cell-prediction.git
   cd cell-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
<!--
3. Optional: Install XGBoost for additional model support:
   ```bash
   # For non-Mac systems:
   pip install xgboost
   
   # For Mac systems, you may need to install OpenMP first:
   brew install libomp
   pip install xgboost
   ```

   Note: The package will work without XGBoost, but will only use Linear Regression and Random Forest models.
-->

## Usage

### Basic Usage

Run the prediction pipeline with training and test data:

```bash
python main.py --train path/to/train_data.csv --test path/to/test_data.csv
```

The results will be saved in the `output` directory by default.

### Command-Line Options

- `--train`: Path to training data file (required)
- `--test`: Path to test data file (optional)
- `--output`: Output directory (default: "output")
- `--no-tune`: Skip hyperparameter tuning
- `--verbose`: Enable verbose output

### Example

```bash
python main.py --train data/train_cells.csv --test data/test_cells.csv --output results --verbose
```

## Input Data Format

The package expects CSV files with the following columns:
- `Donor ID`: Identifier for each donor
- `cluster`: Cluster identifier for cell groups
- `mean_expression`: Array of expression values in string format (e.g., `[0.1, 0.2, -0.5, 0.8, -0.3, 0.7]`)
- `cell_count`: Number of cells
- `percent_cells`: Percentage of cells
- `Cognitive Status`: Target variable (e.g., "Dementia", "No dementia")

## Feature Engineering

The package performs specialized feature engineering for cell expression data:

1. **Cluster Expression Combination**: For each donor, features are created by combining expressions across all clusters
2. **Missing Cluster Handling**: For donors missing certain clusters, global mean values are used as replacements
3. **Comprehensive Feature Set**: Creates a fixed-length feature vector (10 clusters × 6 genes = 60 features) for all donors
4. **Additional Statistics**: Includes cluster proportions and cell count statistics

## Output

The pipeline generates the following outputs in the specified output directory:

- `models/`: Saved model files
- `results/`: 
  - `results_summary.json`: Summary of model performance
  - `test_predictions.csv`: Predictions on test data
- `figures/`: 
  - Feature distribution plots
  - Correlation matrices
  - Confusion matrices
  - ROC curves
  - Feature importance plots
  - Model comparison plots
- `logs/`: Log files of pipeline execution

## Package Structure

```
cell_prediction/
│
├── main.py                     # Main script to run the pipeline
├── config.py                   # Configuration parameters
├── requirements.txt            # Package dependencies
│
├── data/                       # Data handling
│   ├── data_loader.py          # Load and preprocess data
│   └── data_splitter.py        # Split data into train/validation sets
│
├── models/                     # Model implementations
│   ├── base_model.py           # Abstract base model class
│   ├── linear_regression.py    # Linear Regression implementation
│   ├── random_forest.py        # Random Forest implementation
│   ├── xgboost_model.py        # XGBoost implementation (optional)
│   └── model_factory.py        # Factory to get appropriate model
│
├── evaluation/                 # Model evaluation
│   ├── metrics.py              # Evaluation metrics
│   └── validator.py            # Cross-validation implementation
│
├── visualization/              # Visualization tools
│   └── visualizer.py           # Visualization functions
│
└── utils/                      # Utility functions
    └── helpers.py              # Utility functions
```

## Troubleshooting

### Feature Generation Issues

If you encounter issues with feature generation, check:
- The data format matches the expected format
- Each donor has at least one cluster
- The mean_expression arrays are consistent in length

### Class Imbalance

For datasets with few samples or class imbalance, the pipeline automatically:
- Checks class distribution after aggregation
- Warns about classes with too few samples
- Falls back to random splitting when stratified splitting isn't possible

### XGBoost Issues on macOS

If you encounter issues with XGBoost on macOS related to OpenMP libraries, you have several options:

1. Install the OpenMP library with Homebrew and reinstall XGBoost:
   ```bash
   brew install libomp
   pip uninstall xgboost -y
   pip install xgboost
   ```

2. Set environment variables before installation:
   ```bash
   export CPPFLAGS="-Xpreprocessor -fopenmp"
   export CFLAGS="-I/opt/homebrew/opt/libomp/include"
   export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"
   export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"
   pip install xgboost
   ```

3. Run the package without XGBoost - it's designed to work with just Linear Regression and Random Forest models if XGBoost is not available.

## Customization

### Modifying Configuration

You can customize the package behavior by modifying parameters in `config.py`:

- Model hyperparameters
- Cross-validation settings
- Training/validation split ratio
- Output directory names

### Adding New Models

To add a new model:

1. Create a new model file in the `models/` directory
2. Extend the `BaseModel` class
3. Implement required methods (build, fit, predict)
4. Update `model_factory.py` to include your new model

### Customizing Feature Engineering

To modify how features are generated, edit the `aggregate_by_donor` method in `data_loader.py`:

- Change how cluster expressions are combined
- Modify handling of missing clusters
- Add additional statistical features

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.