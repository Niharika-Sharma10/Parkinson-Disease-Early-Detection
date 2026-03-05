import os
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.image_preprocess import ImagePreprocessor
from models.cnn_model import build_transfer_model


def main():

    BASE_PATH = r"C:\Users\DELL\OneDrive\Documents\Parkinson-Disease-Early-Detection\Data\raw\drawings\wave"

    processor = ImagePreprocessor(img_size=224)

    X = []
    y = []

    def load_images(folder_path, label):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                img = processor.preprocess(img_path)
                X.append(img)
                y.append(label)

    load_images(os.path.join(BASE_PATH, "training", "healthy"), 0)
    load_images(os.path.join(BASE_PATH, "training", "parkinson"), 1)

    X = np.array(X)
    y = np.array(y)

    print("Total Wave Images:", len(X))
    print("Shape of X:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_transfer_model()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=8
    )

    model.save("models/saved_models/cnn_wave.keras")

    print("Wave training complete.")


if __name__ == "__main__":
    main()