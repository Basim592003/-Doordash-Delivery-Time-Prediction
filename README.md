# -Doordash-Delivery-Time-Prediction
Overview
This project focuses on predicting the total delivery time for Doordash orders using various machine learning algorithms. It leverages advanced feature engineering, correlation analysis, and multicollinearity handling techniques to identify and train the most effective model for the task. The primary goal is to develop a robust and interpretable prediction pipeline that can estimate delivery times with high accuracy.

Project Workflow
1. Data Preparation
Loaded and cleaned the dataset containing order details, preparation times, and delivery durations.
Handled missing values, removed outliers, and performed basic exploratory data analysis (EDA).
2. Feature Engineering
Created derived features, such as the sum of preparation and delivery times.
Evaluated feature importance using statistical measures like Variance Inflation Factor (VIF) and Gini Importance.
Removed highly correlated and redundant features using correlation heatmaps and VIF analysis.
3. Data Scaling
Standardized numerical features using scalers (e.g., StandardScaler) to ensure compatibility with machine learning models.
4. Machine Learning Models
Implemented and evaluated multiple regression models, including:
Ridge Regression
Decision Trees
Random Forest
XGBoost
LightGBM
Multi-layer Perceptron (MLP)
ANN (Artificial Neural Network)
Conducted hyperparameter tuning for optimal model performance.
5. Model Evaluation
Used metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² (Coefficient of Determination) to compare model performance.
Split the data into training and testing sets to ensure evaluation on unseen data.
Results
The Ridge Regression model demonstrated the best balance between accuracy and simplicity, achieving:

RMSE: 217.38 seconds
MAE: 177.56 seconds
R²: 98.93%
These metrics suggest that the model effectively captures the variability in delivery times, though additional validation techniques (e.g., cross-validation) could further verify robustness.

Technologies Used
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Statsmodels, XGBoost, LightGBM, TensorFlow/Keras.
Key Insights
Feature Importance: Effective feature selection and multicollinearity removal significantly improved model performance and interpretability.
Scaling Matters: Proper data scaling was crucial for algorithms like Ridge Regression and neural networks.
Ridge Regression: Regularization helped handle multicollinearity, leading to a well-performing and generalizable model.
Visual Insights: Correlation heatmaps and feature importance plots offered intuitive insights into the dataset's structure.
Future Work
Explore additional features such as real-time traffic data or weather conditions.
Implement cross-validation to ensure robustness across different subsets of data.
Extend the pipeline to predict delivery delays or optimize driver assignments.
