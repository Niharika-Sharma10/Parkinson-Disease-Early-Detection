import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224

spiral_model = load_model("models/saved_models/cnn_spiral.keras")
wave_model = load_model("models/saved_models/cnn_wave.keras")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_parkinson(spiral_path, wave_path):
    spiral_img = preprocess_image(spiral_path)
    wave_img = preprocess_image(wave_path)

    spiral_prob = spiral_model.predict(spiral_img)[0][0]
    wave_prob = wave_model.predict(wave_img)[0][0]

    final_prob = (spiral_prob + wave_prob) / 2

    if final_prob < 0.33:
        risk = "Low"
    elif final_prob < 0.66:
        risk = "Medium"
    else:
        risk = "High"

    diagnosis = "Parkinson" if final_prob > 0.5 else "Healthy"

    return {
        "spiral_prob": float(spiral_prob),
        "wave_prob": float(wave_prob),
        "final_prob": float(final_prob),
        "diagnosis": diagnosis,
        "risk_level": risk
    }