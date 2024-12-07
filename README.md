# Comparison Of ML Models
This repository contains a Jupyter Notebook that demonstrates how to improve and evaluate multiple regression models for predicting house prices using a real estate dataset. The workflow includes data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset Overview
The dataset consists of the following features:
- **Transaction date**: The date of the transaction (converted to year and month).
- **House age**: The age of the house in years.
- **Distance to the nearest MRT station**: Distance to the nearest public transport station in meters.
- **Number of convenience stores**: Number of stores within walking distance.
- **Latitude**: Geographical coordinate.
- **Longitude**: Geographical coordinate.
- **House price of unit area**: The target variable, representing house price per unit area.

## Workflow

### 1. Data Preprocessing
- Converted `Transaction date` into numerical features: `Transaction year` and `Transaction month`.
- Standardized features using `StandardScaler` for better model performance.

### 2. Model Training
Trained the following regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 3. Model Evaluation
Each model was evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² (coefficient of determination)**

### 4. Hyperparameter Tuning
Performed hyperparameter tuning for the Random Forest model using `GridSearchCV`. The best model was selected based on cross-validated performance.

### 5. Feature Importance
Analyzed the feature importance of the best Random Forest model to understand the contribution of each feature to the predictions.

### 6. Visualizations
- Correlation heatmap to visualize relationships between features.
- Scatter plot comparing actual vs. predicted house prices.

### 7. Model Deployment
The best-performing model was saved as a pickle file (`best_random_forest_model.pkl`) using `joblib` for deployment.

## Usage
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook**:
   Open the Jupyter Notebook and execute the cells step-by-step.

3. **View results**:
   - Model performance metrics will be displayed in tabular form.
   - Visualizations will appear inline within the notebook.

4. **Use the saved model**:
   Load the saved model for making predictions:
   ```python
   import joblib
   model = joblib.load('best_random_forest_model.pkl')
   predictions = model.predict(new_data)
   ```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## File Structure
- `Real_Estate.csv`: The dataset used for training and evaluation.
- `model_improvements.ipynb`: Jupyter Notebook containing the code and analysis.
- `best_random_forest_model.pkl`: Saved Random Forest model.
- `requirements.txt`: List of dependencies.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the creators of the real estate dataset used in this project. This analysis was inspired by the goal of improving machine learning models and gaining deeper insights into feature importance.
