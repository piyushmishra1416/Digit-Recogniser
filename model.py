# Piyush - 2021BEC0023
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import pickle


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train SVM classifier
svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
svm_clf.fit(X_train, y_train)

# Save the trained SVM model to disk
with open('model.pkl', 'wb') as f:
    pickle.dump(svm_clf, f)

# Load the model from disk (optional, to demonstrate model loading)
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Predictions on the test set using the loaded model
y_pred = loaded_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
