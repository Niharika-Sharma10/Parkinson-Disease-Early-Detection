import cv2
import numpy as np
import os


class ImagePreprocessor:
    def __init__(self, img_size=224):   # Changed to 224
        self.img_size = img_size

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def resize(self, img):
        return cv2.resize(img, (self.img_size, self.img_size))

    def normalize(self, img):
        return img.astype(np.float32) / 255.0

    def preprocess(self, image_path):
        img = self.load_image(image_path)
        img = self.resize(img)
        img = self.normalize(img)

        return img


if __name__ == "__main__":
    test_image_path = "Data/raw/drawings/spiral/training/healthy/V01HE02.png"

    processor = ImagePreprocessor(img_size=224)
    processed_img = processor.preprocess(test_image_path)

    print("Image shape:", processed_img.shape)
    print("Min value:", processed_img.min())
    print("Max value:", processed_img.max())