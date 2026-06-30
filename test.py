import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ---------------------------
# 1. Parameters
# ---------------------------
IMG_SIZE = 128
TEST_PATH = "dataset/newTest"    # path for test images + masks
MODEL_PATH = "attention_unet.pth"  # Use .pth instead of .h5 for PyTorch models
BATCH_SIZE = 4

# ---------------------------
# 2. Dataset Loader
# ---------------------------
class UltrasoundDataset(Dataset):
    def __init__(self, folder, img_size=128):
        self.img_size = img_size
        all_files = os.listdir(folder)
        self.image_files = [f for f in all_files if f.endswith(".tif") and "_mask" not in f]
        self.folder = folder

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = img_file.replace(".tif", "_mask.tif")

        img = cv2.imread(os.path.join(self.folder, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.folder, mask_file), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

test_dataset = UltrasoundDataset(TEST_PATH, IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded {len(test_dataset)} test samples.")

# ---------------------------
# 3. Model Load
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(device)

# Fix for PyTorch 2.6+
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
except TypeError:
    # Fallback if PyTorch version enforces weights_only=True by default
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model.load_state_dict(state_dict)
model.eval()
print("Loaded trained model for testing.")

# ---------------------------
# 4. Evaluation
# ---------------------------
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

test_loss, test_dice = 0, 0
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        test_loss += combined_loss(outputs, masks).item()
        test_dice += dice_coeff(outputs, masks).item()

avg_test_loss = test_loss / len(test_loader)
avg_test_dice = test_dice / len(test_loader)

print(f"\n Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}")

# ---------------------------
# 5. Visualization
# ---------------------------
def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        img, mask = dataset[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).cpu().squeeze().numpy()

        axes[i,0].imshow(img.cpu().squeeze(), cmap="gray")
        axes[i,0].set_title("Image")
        axes[i,1].imshow(mask.squeeze(), cmap="gray")
        axes[i,1].set_title("Ground Truth")
        axes[i,2].imshow(pred, cmap="gray")
        axes[i,2].set_title("Prediction")
    plt.show()

# Example usage
visualize_predictions(model, test_dataset, num_samples=3)
