import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load label mapping from JSON
with open('Class_Label.json', 'r') as f:
    labels = json.load(f)

# Function to predict fish type
def predict_fish(img, model, model_name):
    pred = model.predict(img)
    pred_class = np.argmax(pred)
    predicted_fish = labels[str(pred_class)]
    return model_name, predicted_fish

# Paths to saved models
model_paths = {
    'Custom CNN': 'Custom CNN/custom_cnn_model.h5',
    'VGG16': 'VGG16_final.h5',
    'ResNet50': 'ResNet50_final.h5',
    'MobileNet': 'MobileNet_final.h5',
    'InceptionV3': 'InceptionV3_final.h5',
    'EfficientNetV2B0': 'EfficientNetV2B0_final.h5'
}

# Load and preprocess test image
test_img_path = r'G:/My Drive/Data science/Multiclass_Fish_Image_Classification/data/test/animal fish/0AKFISD3OVLE.jpg'

img = image.load_img(test_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0


# Load all models once
models = {name: load_model(path) for name, path in model_paths.items()}

# Predict and display results
print("\nPredictions for test image:\n")
print(f"{'Model':<20} | {'Predicted Fish'}")
print("-"*40)
for name, model in models.items():
    _, prediction = predict_fish(img_array, model, name)
    print(f"{name:<20} | {prediction}")
