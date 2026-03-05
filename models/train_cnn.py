import os
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.image_preprocess import ImagePreprocessor
from models.cnn_model import CNNModel

# Base dataset path
BASE_PATH = r"C:\Users\DELL\OneDrive\Documents\Parkinson-Disease-Early-Detection\Data\raw\drawings"

processor = ImagePreprocessor(img_size=128)

X = []
y = []

def load_images_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = processor.preprocess(img_path)
            X.append(img)
            y.append(label)

# Load spiral training data
load_images_from_folder(os.path.join(BASE_PATH, "spiral", "training", "healthy"), 0)
load_images_from_folder(os.path.join(BASE_PATH, "spiral", "training", "parkinson"), 1)

# Load wave training data
load_images_from_folder(os.path.join(BASE_PATH, "wave", "training", "healthy"), 0)
load_images_from_folder(os.path.join(BASE_PATH, "wave", "training", "parkinson"), 1)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
cnn = CNNModel()

# Train model
history = cnn.train(X_train, y_train, X_val, y_val, epochs=15, batch_size=16)

# Save model
cnn.save_model()

print("Training complete.")