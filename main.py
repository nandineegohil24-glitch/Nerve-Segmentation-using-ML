import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Parameters
IMG_SIZE = 128
DATASET_PATH = "dataset/train"

# Load images and corresponding masks
X = []
y = []

all_files = os.listdir(DATASET_PATH)
image_files = [f for f in all_files if f.endswith(".tif") and "_mask" not in f]

for img_file in image_files:
    mask_file = img_file.replace(".tif", "_mask.tif")
    img_path = os.path.join(DATASET_PATH, img_file)
    mask_path = os.path.join(DATASET_PATH, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠ No mask found for {img_file}, skipping...")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠ Could not load {img_file} or its mask, skipping...")
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    X.append(img)
    y.append(mask)

X = np.array(X, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

print(f"✅ Loaded {len(X)} images and masks successfully.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN for segmentation (autoencoder-like)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2), padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2), padding='same'),

    # Decoder
    Conv2D(64, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),

    # Output layer: same size as input, 1 channel for mask
    Conv2D(1, (3,3), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
# Save model
model.save("ann_segmentation.h5")
print("✅ ANN trained and saved as unet_segmentation.h5")