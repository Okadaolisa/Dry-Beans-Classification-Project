# Dry Beans Classification Project

## Overview
The goal of this project is to develop a system for the automatic detection of seven types of dry bean seeds using data captured with a high-resolution camera. By leveraging machine learning methods, this project aims to accurately classify the bean types based on a set of extracted features.

## Dataset
- **Features:** 16 numerical attributes extracted from images of beans.
- **Labels:** The dataset contains the following types of beans:
  1. Seker
  2. Barbunya
  3. Bombay
  4. Cali
  5. Dermosan
  6. Horoz
  7. Sira

The dataset was provided as a CSV file and processed using Python for machine learning tasks.

## Phases of Machine Learning
1. **Training:**
   - Subset of data is used to train the model to learn the input-output relationship.
2. **Testing:**
   - Evaluate the trained model on a separate subset of data.
3. **Application:**
   - Use the trained model to make predictions on new data in real-world scenarios.

## Implementation Steps

### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
```

### 2. Load the Dataset
```python
# Load dataset from an Excel file
dataset_df = pd.read_excel('Dry_Bean_Dataset.xlsx')

# Display the first few rows of the dataset
dataset_df.head()
```

### 3. Preprocessing
```python
# Extract features and target labels
X = dataset_df.iloc[:, :-1]  # All columns except the last
y = dataset_df.iloc[:, -1]  # The last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training and Evaluation
```python
# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 5. Visualizations (Optional)
```python
# Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances, align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title('Feature Importance in Random Forest Model')
plt.show()
```

## Results
- **Accuracy:** Achieved a high classification accuracy on the test set.
- **Insights:** The most important features contributing to bean classification were identified.

## Conclusion
This project demonstrates the application of machine learning in agricultural classification tasks. By utilizing features extracted from images, we successfully classified dry bean types with high accuracy.
