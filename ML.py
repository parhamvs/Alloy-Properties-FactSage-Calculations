# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import shap

# Install required packages if not already installed
# !pip install pdpbox
# !pip install shap

# Mount Google Drive if using Google Colab
# from google.colab import drive
# drive.mount('/content/gdrive')

# Load the dataset
# Assuming 'Mg-GA-Added.xlsx' is in the current directory
df = pd.read_excel('Mg-GA-Added.xlsx')

# Feature Engineering
# Sum various columns to create new features
Sum_column_Sr = df['Sr2Mg17(s)'] + df['Sr5Si3(s)'] + df['SrZn5(s)'] + df['Sr2Zn43Mg55(s)']
Sum_column_Zn = df['Mg12Zn13(s)'] + df['Ca2Mg55Zn43(s)'] + df['SrZn5(s)'] + df['Sr2Zn43Mg55(s)'] + df['YZnMg12(s)'] + df['Zn2Zr(s)'] + df['ZnZr(s)'] + df['Nd3Zn22(s)'] + df['NdZn2Mg(s)'] + df['Nd2Zn9Mg5(s)'] + df['NdZn2Al2(s)'] + df['GdZnMg12(s)']
Sum_column_Nd = df['Nd5Mg41(s)'] + df['Nd3Al11(s)'] + df['Nd3Zn22(s)'] + df['NdZn2Mg(s)'] + df['Nd2Zn9Mg5(s)'] + df['NdZn2Al2(s)']
Sum_column_Ca = df['Mg2Ca(s)'] + df['Al2Ca(s)'] + df['CaMgSi(s)'] + df['Mn2CaAl10(s)'] + df['Ca2Mg55Zn43(s)']
Sum_column_Y = df['Y(s)'] + df['AlY(s)'] + df['Al3Y(s)'] + df['Al2Y3(s)'] + df['Y6Mn23(s)'] + df['YZnMg12(s)']
Sum_column_Mn = df['Mn(s)'] + df['Al4Mn(s)'] + df['Al11Mn4(s)'] + df['Mn3Si(s)'] + df['Mn5Si3(s)'] + df['Mn2CaAl10(s)'] + df['Al7CuMn2(s)'] + df['Y6Mn23(s)'] + df['Mn2Zr(s)']
Sum_column_Zr = df['Zr(s)'] + df['Al3Zr(s)'] + df['Mn2Zr(s)'] + df['ZnZr(s)'] + df['Zn2Zr(s)']
Sum_column_Al = df['Al30Mg23(s)'] + df['Al2Ca(s)'] + df['Mn2CaAl10(s)'] + df['Al7Cu3Mg6(s)'] + df['Al5Cu6Mg2(s)'] + df['Al7CuMn2(s)'] + df['AlY(s)'] + df['Al3Y(s)'] + df['Al2Y3(s)'] + df['Al3Zr(s)'] + df['Nd3Al11(s)'] + df['NdZn2Al2(s)']

# Create new columns in the DataFrame
df['Alkline'] = Sum_column_Sr + Sum_column_Ca
df['Transition'] = Sum_column_Zr + Sum_column_Y + Sum_column_Mn + Sum_column_Zn + Sum_column_Cu
df['Lanthanides'] = Sum_column_Nd + Sum_column_Gd
df['Post-transition'] = Sum_column_Al
df['Metalloids'] = Sum_column_Si
df['SUM'] = df['Alkline'] + df['Transition'] + df['Lanthanides'] + df['Post-transition'] + df['Metalloids']

# Drop unnecessary columns
df = df.drop(df.iloc[:, 13:53], axis=1)

# Encode categorical variables if needed
# Example: df = pd.get_dummies(df, columns=['Process'], prefix='Process')

# Define target variables and encode if needed
# Example: y = pd.get_dummies(df['Heat Treatment'], prefix='Heat Treatment')

# Drop columns not needed for modeling
# Example: df = df.drop(['Heat Treatment', 'Mg', 'Heat Treatment_F'], axis=1)

# Separate target variable from features
# Example: X = df.drop(target_column, axis=1)
# Example: y = df[target_column]

# Split the dataset into training and testing sets
# Example: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Create a Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=8, random_state=33)

# Train the model on the training data
rf.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = rf.predict(X_test)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics
print(f'R-squared Score: {r2:.2f}')
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Plot permutation importance
perm_importance = permutation_importance(rf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 10))
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation Importance for YS")
plt.show()

# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.xlabel("SHAP Value (Impact on Model Output)")
plt.title("SHAP Summary Plot for YS")
plt.show()
