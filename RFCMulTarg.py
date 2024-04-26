import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

# Load your data
data1 = pd.read_csv('featuresFemHel.csv')
data2 = pd.read_csv('featuresFemUnHel.csv')
data3 = pd.read_csv('featuresMalHel.csv')
data4 = pd.read_csv('featuresMalUnHel.csv')

# Concatenate the dataframes vertically (along rows)
combined_data = pd.concat([data1, data2, data3, data4], ignore_index=True)

# Separate features (X) and labels (y)
X = combined_data.iloc[:, 3:]  # Exclude first three columns (File_name, GENDER, HEALTH)
y = combined_data['HEALTH']

# Convert the labels (y) to Healthy and Unhealthy categories
label_mapping = {'Healthy': 0, 'Unhealthy': 1}
combined_data['HEALTH_CATEGORIZED'] = combined_data['HEALTH'].map(label_mapping)
y_categorized = combined_data['HEALTH_CATEGORIZED']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorized, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Classification report
report = classification_report(y_test, y_pred_test, target_names=['Healthy', 'Unhealthy'])

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Classification Report:\n", report)