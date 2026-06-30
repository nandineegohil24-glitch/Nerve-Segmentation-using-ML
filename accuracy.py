import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# -------------------------------
# 1️⃣ Paths
# -------------------------------
DATA_DIR = r"D:\ML-Ultrasound\dataset\newTest"
MODEL_PATH = r"D:\ML-Ultrasound\attention_unet.h5"
IMG_SIZE = 128   # change if your model uses a different size

# -------------------------------
# 2️⃣ Load Model
# -------------------------------
print("🔹 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!- ", MODEL_PATH)

# -------------------------------
# 3️⃣ Load Images & Masks
# -------------------------------
images, masks = [], []

for file in os.listdir(DATA_DIR):
    if file.endswith(".tif") and "_mask" not in file:
        img_path = os.path.join(DATA_DIR, file)
        mask_path = os.path.join(DATA_DIR, file.replace(".tif", "_mask.tif"))

        if not os.path.exists(mask_path):
            continue

        # Load grayscale images
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

        # Expand dims for (H,W,1)
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

X = np.array(images)
Y = np.array(masks)
print(f" Loaded {len(X)} image-mask pairs.")

# -------------------------------
# 4️⃣ Predict
# -------------------------------
print(" Running predictions...")
preds = model.predict(X, verbose=1)
preds = (preds > 0.5).astype(np.uint8)

# -------------------------------
# 5️⃣ Metrics Calculation
# -------------------------------
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

Y_flat = (Y > 0.5).astype(np.uint8).flatten()
P_flat = (preds > 0.5).astype(np.uint8).flatten()


precision = precision_score(Y_flat, P_flat)
recall = recall_score(Y_flat, P_flat)
f1 = f1_score(Y_flat, P_flat)
accuracy = accuracy_score(Y_flat, P_flat)
iou = iou_score(Y_flat, P_flat)
dice = dice_score(Y_flat, P_flat)

print("\n Evaluation Metrics:")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Dice Coef : {dice:.4f}")
