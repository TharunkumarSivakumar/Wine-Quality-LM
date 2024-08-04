import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from warnings import filterwarnings

# Suppress warnings for cleaner output
filterwarnings(action='ignore')

# Load the wine quality dataset
# Make sure the 'winequality-red.csv' file is in the same directory as this script
wine = pd.read_csv("winequality-red.csv")
print("Successfully Imported Data!")
print(wine.head())

# Display the shape of the dataset
print(wine.shape)

# Display summary statistics
print(wine.describe(include='all'))

# Check for missing values
print(wine.isna().sum())

# Display the correlation matrix
print(wine.corr())

# Group by wine quality and display the mean of each feature
print(wine.groupby('quality').mean())

# Data Visualization
# Countplot of wine quality
sns.countplot(wine['quality'])
plt.show()

# Countplot of pH
sns.countplot(wine['pH'])
plt.show()

# Countplot of alcohol
sns.countplot(wine['alcohol'])
plt.show()

# Countplot of fixed acidity
sns.countplot(wine['fixed acidity'])
plt.show()

# Countplot of volatile acidity
sns.countplot(wine['volatile acidity'])
plt.show()

# Countplot of citric acid
sns.countplot(wine['citric acid'])
plt.show()

# Countplot of density
sns.countplot(wine['density'])
plt.show()

# KDE plot of quality
sns.kdeplot(wine.query('quality > 2').quality)
plt.show()

# Distribution plot of alcohol
sns.distplot(wine['alcohol'])
plt.show()

# Box plot of all features
wine.plot(kind='box', subplots=True, layout=(4,4), sharex=False)
plt.show()

# Density plot of all features
wine.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()

# Histogram of all features
wine.hist(figsize=(10,10), bins=50)
plt.show()

# Heatmap of the correlation matrix
corr = wine.corr()
sns.heatmap(corr, annot=True)
plt.show()

# Pairplot of the dataset
sns.pairplot(wine)
plt.show()

# Violin plot of quality vs alcohol
sns.violinplot(x='quality', y='alcohol', data=wine)
plt.show()

# Create a binary target variable for classification (good quality: quality >= 7)
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]

# Separate feature variables and target variable
X = wine.drop(['quality', 'goodquality'], axis=1)
Y = wine['goodquality']

# Display the proportion of good vs bad wines
print(wine['goodquality'].value_counts())

# Feature Importance using ExtraTreesClassifier
classifier = ExtraTreesClassifier()
classifier.fit(X, Y)
score = classifier.feature_importances_
print(score)

# Plot the feature importances
importances = pd.Series(classifier.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values()

plt.figure(figsize=(10, 6))
importances_sorted.plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
