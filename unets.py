import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import albumentations as A  # ✅ Augmentation library

# ---------------------------
# 1. Parameters
# ---------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset/train"
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
    mask = (mask > 127).astype(np.float32)

    X.append(img)
    y.append(mask)

X = np.array(X, dtype=np.float32).reshape(-1, 1, IMG_SIZE, IMG_SIZE) / 255.0
y = np.array(y, dtype=np.float32)

print(f"✅ Loaded {len(X)} images and masks successfully.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. Dataset with Augmentation
# ---------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(p=0.2),
])

val_transform = A.Compose([])  # No augmentation for validation

class UltrasoundDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx][0]  # remove channel for albumentations
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = np.expand_dims(img, axis=0)  # add channel back
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

train_dataset = UltrasoundDataset(X_train, y_train, transform=train_transform)
val_dataset = UltrasoundDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# 4. Model Setup (same)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
).to(device)

# ---------------------------
# 5. Loss, Optimizer, Dice metric (same)
# ---------------------------
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ---------------------------
# 6. Training Loop (same)
# ---------------------------
best_dice = 0
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), f"best_unet_{timestamp}.pth")
        print("✅ Saved best model")

# ---------------------------
# 7. Visualization (same)
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

visualize_predictions(model, X_val, y_val, num_samples=3)
