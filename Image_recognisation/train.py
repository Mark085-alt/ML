import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Image size and paths
IMG_SIZE = (150, 150)
TRAIN_DIR = r"D:\College Project\Combined\Image_recognisation\dataset\train"
VAL_DIR = r"D:\College Project\Combined\Image_recognisation\dataset\val"
TEST_DIR = r"D:\College Project\Combined\Image_recognisation\dataset\test_images"

# Load training and validation data
datagen = ImageDataGenerator(rescale=1.0/255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=4,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=4,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("\nTraining the model...")
model.fit(train_data, validation_data=val_data, epochs=5)

# Get class labels (should be ['cat', 'dog'] based on folder names)
class_labels = list(train_data.class_indices.keys())

# Predict test images
print("\nTesting on new images:")
for filename in os.listdir(TEST_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(TEST_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {filename}")
            continue

        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_input)[0][0]
        predicted_label = class_labels[1] if prediction > 0.5 else class_labels[0]
        is_cat = predicted_label == 'cat'

        print(f"Image: {filename} --> {is_cat}")
