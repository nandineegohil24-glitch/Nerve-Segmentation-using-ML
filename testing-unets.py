import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# ---------------------------
# 1. Parameters
# ---------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset/train"   # change if needed
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

# ---------------------------
# 2. Load Dataset
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

    mask = (mask > 127).astype(np.float32)  # ✅ binarize

    X.append(img)
    y.append(mask)

X = np.array(X, dtype=np.float32).reshape(-1, 1, IMG_SIZE, IMG_SIZE) / 255.0
y = np.array(y, dtype=np.float32)

print(f"✅ Loaded {len(X)} images and masks successfully.")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. Dataset Wrapper
# ---------------------------
class UltrasoundDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx].unsqueeze(0)  # ✅ add channel dim

train_dataset = UltrasoundDataset(X_train, y_train)
val_dataset = UltrasoundDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# 4. Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,   # binary segmentation
).to(device)

# ---------------------------
# 5. Loss & Optimizer
# ---------------------------
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Dice metric
def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ---------------------------
# 6. Training Loop
# ---------------------------
best_dice = 0
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("saved_models", exist_ok=True)  # folder for models

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    train_loss, train_dice = 0, 0

    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coeff(outputs, masks).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)

    # Validation
    model.eval()
    val_loss, val_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_coeff(outputs, masks).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    print(f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

    # ✅ Save every epoch
    model_path = f"saved_models/unet_epoch{epoch+1:02d}_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved model at {model_path}")

    # ✅ Save best model separately
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), f"saved_models/unet_best_{timestamp}.pth")
        print("🏆 Saved best model")

# ---------------------------
# 7. Visualize Predictions
# ---------------------------
def visualize_predictions(model, X_val, y_val, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        img = torch.tensor(X_val[i:i+1], dtype=torch.float32).to(device)
        mask = y_val[i]
        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).cpu().squeeze().numpy()

        axes[i,0].imshow(X_val[i,0], cmap="gray")
        axes[i,0].set_title("Image")
        axes[i,1].imshow(mask, cmap="gray")
        axes[i,1].set_title("Ground Truth")
        axes[i,2].imshow(pred, cmap="gray")
        axes[i,2].set_title("Prediction")
    plt.show()

# ---------------------------
# 8. Load and Test Saved Model
# ---------------------------
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {model_path}")
    return model

# Example usage: load a specific epoch model
# model_epoch10 = load_model("saved_models/unet_epoch10_20250917_xxxxx.pth")
# visualize_predictions(model_epoch10, X_val, y_val, num_samples=3)
