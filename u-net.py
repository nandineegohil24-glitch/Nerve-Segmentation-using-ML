import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Parameters
IMG_SIZE = 128
DATASET_PATH = "dataset/train"

# ---------------------------
# 1. Load images and masks
# ---------------------------
X, y = [], []
all_files = os.listdir(DATASET_PATH)
image_files = [f for f in all_files if f.endswith(".tif") and "_mask" not in f]

for img_file in image_files:
    mask_file = img_file.replace(".tif", "_mask.tif")
    img_path = os.path.join(DATASET_PATH, img_file)
    mask_path = os.path.join(DATASET_PATH, mask_file)

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    X.append(img)
    y.append(mask)

X = np.array(X, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

print(f"✅ Loaded {len(X)} images and masks successfully.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 2. Build U-Net Model
# ---------------------------
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2,2))(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2,2))(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# 3. Train Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=20
)

# ---------------------------
# 4. Evaluate with Dice & IoU
# ---------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred.flatten() > 0.5).astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred.flatten() > 0.5).astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

preds = model.predict(X_val)
print("Dice:", dice_coefficient(y_val, preds))
print("IoU:", iou_score(y_val, preds))

# ---------------------------
# 5. Show Example Overlay
# ---------------------------
idx = 0
test_img = (X_val[idx,:,:,0] * 255).astype(np.uint8)
true_mask = (y_val[idx,:,:,0] * 255).astype(np.uint8)
pred_mask = (preds[idx,:,:,0] > 0.5).astype(np.uint8) * 255

# Overlay red prediction on image
overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
overlay[pred_mask==255] = [255, 0, 0]  # red

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(test_img, cmap='gray'); plt.title("Ultrasound")
plt.subplot(1,3,2); plt.imshow(true_mask, cmap='gray'); plt.title("Ground Truth")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Prediction Overlay")
plt.show()

# Save model
model.save("unet_segmentation.h5")
print("✅ U-Net trained and saved as unet_segmentation.h5")
