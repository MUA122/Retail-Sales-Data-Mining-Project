Retail Sales Data Mining Project - Split Files

Files:
1. preprocessing.py
   - Loads dataset
   - Shows raw data details
   - Checks missing values, unique values, duplicates, data types
   - Cleans data
   - Creates Month, Month Number, Year, Age Group, High Spender

2. kmeans_model.py
   - Trains K-Means customer segmentation
   - Shows Inertia, Silhouette Score, Calinski Harabasz Score, Davies Bouldin Score
   - Builds business cluster summary and live cluster prediction

3. random_forest_model.py
   - Trains Random Forest High-Spender classifier
   - Trains Random Forest Product Category classifier
   - Shows Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC AUC, confusion matrix, classification report

4. linear_regression_model.py
   - Trains Linear Regression to predict Total Amount
   - Shows R2, Explained Variance, MSE, RMSE, MAE, Mean Error, Median Absolute Error, Max Error
   - Shows regression coefficients and prediction result table

5. streamlit_app.py
   - Final dashboard app
   - Imports all previous files
   - Contains Overview, Preprocessing, Owner Dashboard, K-Means, Random Forest, Linear Regression, and CRISP-DM tabs

How to run:
1. Put retail_sales_dataset.csv in the same folder.
2. Install requirements:
   pip install -r requirements.txt
3. Run:
   streamlit run streamlit_app.py
