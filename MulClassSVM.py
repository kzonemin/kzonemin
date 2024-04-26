import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier

# Load data from all four CSV files
data1 = pd.read_csv('featuresFemHel.csv')
data2 = pd.read_csv('featuresFemUnHel.csv')
data3 = pd.read_csv('featuresMalHel.csv')
data4 = pd.read_csv('featuresMalUnHel.csv')

# Concatenate all data into a single DataFrame
combined_data = pd.concat([data1, data2, data3, data4], ignore_index=True)

# Separate features (X) and labels (y)
X = combined_data.iloc[:, 3:]  # Exclude first three columns (File_name, GENDER, HEALTH)
y = combined_data['HEALTH'].map({'Healthy': 0, 'Unhealthy': 1})  # Convert HEALTH to binary labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the multi-class SVM model
svm_model = OneVsRestClassifier(SVC(kernel='linear'))
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = svm_model.predict(X_train_scaled)
y_pred_test = svm_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy (Multi-class SVM):", train_accuracy)
print("Testing Accuracy (Multi-class SVM):", test_accuracy)

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_test))