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
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------
# 1. Parameters
# ---------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset/train"   # change if needed
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
PATIENCE = 7
WEIGHT_DECAY = 1e-5

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

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"✅ Loaded {len(X)} images and masks successfully.")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. Augmentation with Albumentations
# ---------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=(0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
    A.GaussianBlur(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# ---------------------------
# 4. Dataset Class
# ---------------------------
class UltrasoundDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

# ---------------------------
# 5. Model, Loss, and Training
# ---------------------------
def train_model():
    # Use CPU (safe for your setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = UltrasoundDataset(X_train, y_train, transform=train_transform)
    val_dataset = UltrasoundDataset(X_val, y_val, transform=val_transform)

    # ✅ num_workers=0, pin_memory=False (for CPU/Windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_dropout=0.3,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def combined_loss(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    def dice_coeff(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    best_dice = 0
    no_improve_epochs = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "new_saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss, train_dice = 0.0, 0.0

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
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coeff(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 LR after scheduler step: {current_lr:.6f}")

        print(f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

        # Save every epoch
        model_path = os.path.join(save_dir, f"unet_epoch{epoch+1:02d}_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_path = os.path.join(save_dir, f"unet_best_{timestamp}.pth")
            torch.save(model.state_dict(), best_path)
            print("🏆 Saved best model:", best_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

    print(f"✅ Training finished. Best validation Dice: {best_dice:.4f}")
    return model

# ---------------------------
# 6. Visualization
# ---------------------------
def visualize_predictions(model, X_val, y_val, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        img = X_val[i]
        mask = y_val[i]
        transformed = val_transform(image=img, mask=mask)
        img_t = transformed['image'].unsqueeze(0)
        mask = transformed['mask'].cpu().numpy()

        with torch.no_grad():
            pred = model(img_t)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).cpu().squeeze().numpy()

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
    plt.show()

# ---------------------------
# 7. Safe entry point (important for Windows)
# ---------------------------
if __name__ == "__main__":
    model = train_model()
    # visualize_predictions(model, X_val, y_val, num_samples=3)
