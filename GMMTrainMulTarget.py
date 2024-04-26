import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
data1 = pd.read_csv('featuresFemHel.csv')
data2 = pd.read_csv('featuresFemUnHel.csv')
data3 = pd.read_csv('featuresMalHel.csv')
data4 = pd.read_csv('featuresMalUnHel.csv')

# Combine all dataframes
combined_data = pd.concat([data1, data2, data3, data4])

# Separate features (X) and labels (y)
X = combined_data.iloc[:, 3:-2]  # Exclude first three columns (File_name, GENDER, HEALTH) and last two columns (Jitter, Shimmer)
y = combined_data['HEALTH']

# Preprocessing - Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)  # You can adjust the number of components as needed
gmm.fit(X_train)

# Predictions
y_pred_train = gmm.predict(X_train)
y_pred_test = gmm.predict(X_test)

# Convert string labels to integers
label_mapping = {'Healthy': 0, 'Unhealthy': 1}
y_train_int = y_train.map(label_mapping)
y_test_int = y_test.map(label_mapping)

# Calculate accuracy
train_accuracy = accuracy_score(y_train_int, y_pred_train)
test_accuracy = accuracy_score(y_test_int, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test_int, y_pred_test, target_names=['Healthy', 'Unhealthy']))