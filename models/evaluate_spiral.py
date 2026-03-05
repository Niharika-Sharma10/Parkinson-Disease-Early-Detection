import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model
model = load_model("models/saved_models/cnn_spiral.keras")

# Path to spiral test folder
test_path = "Data/raw/drawings/spiral/testing"

X_test = []
y_true = []

# Loop through class folders
for label in ["healthy", "parkinson"]:
    class_path = os.path.join(test_path, label)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        
        X_test.append(img)
        
        if label == "healthy":
            y_true.append(0)
        else:
            y_true.append(1)

X_test = np.array(X_test)
y_true = np.array(y_true)

# Predict
pred_probs = model.predict(X_test)
y_pred = (pred_probs > 0.5).astype(int)

# Metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))