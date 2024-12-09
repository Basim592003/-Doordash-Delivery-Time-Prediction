# -Doordash-Delivery-Time-Prediction
Overview
Project Description This project aims to predict the total delivery time for Doordash orders using a number of machine learning algorithms It does this by utilizing advanced feature engineering techniques and correlation analysis as well as handling multicollinearity to determine and train the best model for the job. The main objective is to implement a strong and interpretable prediction pipeline that is capable of estimating delivery times at a high degree of precision.

Project Workflow
1. Data Preparation
Loaded and cleaned the dataset containing order details, preparation times, and delivery durations.
Handled missing values,and performed basic exploratory data analysis (EDA).
2. Feature Engineering
Created derived features like the total of preparation and delivery time.
Importance assessment for feature evaluation using statistical measures like VIF and Gini Importance.
Eliminated highly correlated and redundant features using correlation heatmaps and VIF analysis.
3. Data Scaling
Preprocessed data: Used StandardScaler, MinMax and no scale to compare.
4. Machine Learning Models
Several ML models used in the study include:
Ridge Regression
Decision Trees
Random Forest
XGBoost
LightGBM
Multi-layer Perceptron (MLP)
ANN (Artificial Neural Network)
5. Model Evaluation
Compared the performances of these models using RMSE, MAE, and R².
The data is divided into a training set and a test set to ensure that the evaluation will be done on unseen data.
Results
The Ridge Regression model showed the best balance between accuracy and simplicity, reaching:
RMSE: 217.38 seconds
MAE: 177.56 seconds
R²: 98.93%
These metrics suggest that the model effectively captures the variability in delivery times, though additional validation techniques-e.g., cross-validation-could further verify robustness.
Technologies Used

Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Statsmodels, XGBoost, LightGBM, TensorFlow/Keras.
Key Insights
Feature Importance: Effective feature selection and multicollinearity removal significantly improved model performance and interpretability.
Scaling: Scale factor played a role in model performance.
Ridge Regression: Regularization helped resolve the issue of multicollinearity and allowed the model to perform well and generalize.
Visual Insights: Correlation heatmaps and feature importance plots provided intuitive insights into the structure of the dataset.



