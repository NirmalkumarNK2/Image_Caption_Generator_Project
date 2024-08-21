import cv2
import numpy as np
import cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    return image

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax') 
    ])
    return model

def map_prediction_to_caption(prediction):
    captions = {0: "A scenic landscape", 1: "A cute animal", 2: "A delicious meal"}
    return captions[np.argmax(prediction)]

if __name__ == "__main__":
    image_path = "your_image.jpg"  
    model = create_cnn_model()

    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)

    caption = map_prediction_to_caption(prediction)
    print("Generated Caption:", caption)
