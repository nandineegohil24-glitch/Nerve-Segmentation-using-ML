import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------
# 1. Settings
# ---------------------------
IMG_SIZE = 128
MODEL_PATH = "attention_unet.h5"
TEST_PATH = "dataset/newTest"   # Path where your test images are stored

# ---------------------------
# 2. Load the Trained Model
# ---------------------------
print("🔹 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ---------------------------
# 3. Dice & IoU Metrics
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

# ---------------------------
# 4. Load Test Images
# ---------------------------
print("🔹 Loading test images...")
test_images, test_masks = [], []
all_files = os.listdir(TEST_PATH)
image_files = [f for f in all_files if f.endswith(".tif") and "_mask" not in f]

for img_file in image_files:
    mask_file = img_file.replace(".tif", "_mask.tif")
    img_path = os.path.join(TEST_PATH, img_file)
    mask_path = os.path.join(TEST_PATH, mask_file)

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    test_images.append(img)
    test_masks.append(mask)

test_images = np.array(test_images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
test_masks = np.array(test_masks, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

print(f"✅ Loaded {len(test_images)} test images and masks.")

# ---------------------------
# 5. Predict & Evaluate
# ---------------------------
print("🔹 Predicting on test images...")
preds = model.predict(test_images, verbose=1)

dice = dice_coefficient(test_masks, preds)
iou = iou_score(test_masks, preds)

print(f"✅ Dice Coefficient: {dice:.4f}")
print(f"✅ IoU Score: {iou:.4f}")

# ---------------------------
# 6. Visualize Example
# ---------------------------
idx = 0
test_img = (test_images[idx, :, :, 0] * 255).astype(np.uint8)
true_mask = (test_masks[idx, :, :, 0] * 255).astype(np.uint8)
pred_mask = (preds[idx, :, :, 0] > 0.5).astype(np.uint8) * 255

overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
overlay[pred_mask == 255] = [255, 0, 0]  # red prediction overlay

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_img, cmap='gray')
plt.title("Test Image")

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='gray')
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Prediction Overlay")

plt.tight_layout()
plt.show()
