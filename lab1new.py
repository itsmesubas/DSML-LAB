# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Titanic dataset (replace with your dataset if needed)
data = sns.load_dataset('titanic')

# 1. Overview of the dataset
print("### Dataset Overview ###\n")
print("First 5 Rows of the Dataset:")
print(data.head(), "\n")
print("Dataset Information:")
print(data.info(), "\n")
print("Summary Statistics:")
print(data.describe(include='all'), "\n")

# 2. Checking for missing values
print("### Missing Values ###\n")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0], "\n")

# 3. Visualizing missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# 4. Analyzing the target variable (e.g., 'survived')
print("### Target Variable Analysis ###\n")
print(data['survived'].value_counts(), "\n")
sns.countplot(data=data, x='survived', palette='viridis')
plt.title("Survival Count")
plt.show()

# 5. Analyzing numerical features
numerical_cols = data.select_dtypes(include=[np.number]).columns
print("### Numerical Feature Analysis ###\n")
print("Numerical Columns:", numerical_cols, "\n")

for col in numerical_cols:
    sns.histplot(data[col].dropna(), kde=True, color='blue', bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# 6. Analyzing categorical features
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
print("### Categorical Feature Analysis ###\n")
print("Categorical Columns:", categorical_cols, "\n")

for col in categorical_cols:
    sns.countplot(data=data, y=col, palette='viridis', order=data[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.show()

# 7. Correlation analysis for numerical features
print("### Correlation Analysis ###\n")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 8. Pairplot to visualize relationships
print("### Pairplot of Features ###\n")
sns.pairplot(data, hue='survived', palette='coolwarm', diag_kind='kde')
plt.show()

print("EDA Complete!")
