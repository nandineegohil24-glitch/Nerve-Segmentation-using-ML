# ============================================================
# Scale-Attention U-Net++ (compatible with your tif dataset)
# ============================================================
# Expects files like:
#   dataset/train/1_1.tif
#   dataset/train/1_1_mask.tif
#   1_2.tif
#   1_2_mask.tif
# etc.
#
# Usage:
#   python scale_attention_unetpp.py
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset/train"   # <-- update if needed
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

# ---------------------------
# 1. Load images and masks
# ---------------------------
X, y = [], []

image_files = sorted([f for f in os.listdir(DATASET_PATH)
                      if f.endswith(".tif") and "_mask" not in f])

for img_file in image_files:
    mask_file = img_file.replace(".tif", "_mask.tif")
    img_path = os.path.join(DATASET_PATH, img_file)
    mask_path = os.path.join(DATASET_PATH, mask_file)

    if not os.path.exists(mask_path):
        print(f"⚠️ Missing mask for {img_file}, skipping.")
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

print(f"✅ Loaded {len(X)} image-mask pairs successfully.")

# ---------------------------
# 2. Train / Val split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---------------------------
# 3. tf.data pipeline
# ---------------------------
def augment(image, mask):
    # Random flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # small random rotation
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    return image, mask

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(512, seed=SEED).map(
    lambda im, m: augment(im, m), num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# 4. Helper conv block
# ---------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

# ---------------------------
# 5. Scale Attention block
#    - Resizes skip features to x spatial size (using Keras layers)
#    - Aligns channels with 1x1 conv before addition
# ---------------------------
def scale_attention(x, skip_connections):
    """
    x: decoder feature (Keras tensor)
    skip_connections: list of encoder feature tensors
    returns: multiplied features (x * attention_map)
    """
    att = None
    target_h = int(x.shape[1])
    target_w = int(x.shape[2])
    target_ch = int(x.shape[-1])

    for skip in skip_connections:
        # Resize skip spatial dims to match x (use static shape ints)
        skip_resized = layers.Resizing(height=target_h, width=target_w)(skip)

        # Align channels -> to target_ch
        skip_resized = layers.Conv2D(target_ch, kernel_size=1, padding='same')(skip_resized)

        if att is None:
            att = skip_resized
        else:
            att = layers.Add()([att, skip_resized])

    # activation + channel-wise sigmoid attention
    att = layers.Activation('relu')(att)
    att = layers.Conv2D(target_ch, kernel_size=1, padding='same', activation='sigmoid')(att)

    # apply attention
    return layers.Multiply()([x, att])

# ---------------------------
# 6. Scale-Attention U-Net++ model
#    (a compact U-Net++-like nested decoder using our scale_attention)
# ---------------------------
def scale_attention_unetpp(input_size=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(input_size)

    # Encoder (contracting path)
    c1 = conv_block(inputs, 32)      # 128x128x32
    p1 = layers.MaxPooling2D(2)(c1)  # 64x64x32

    c2 = conv_block(p1, 64)          # 64x64x64
    p2 = layers.MaxPooling2D(2)(c2)  # 32x32x64

    c3 = conv_block(p2, 128)         # 32x32x128
    p3 = layers.MaxPooling2D(2)(c3)  # 16x16x128

    c4 = conv_block(p3, 256)         # 16x16x256
    p4 = layers.MaxPooling2D(2)(c4)  # 8x8x256

    c5 = conv_block(p4, 512)         # bottleneck 8x8x512

    # Decoder with nested attention (rough U-Net++ style)
    # Level 4 decoder
    u4 = layers.UpSampling2D(2)(c5)               # 16x16x512
    # Attention using c4, c3, c2 (multi-scale)
    u4 = scale_attention(u4, [c4, c3, c2])
    u4 = conv_block(u4, 256)                      # 16x16x256

    # Level 3 decoder
    u3 = layers.UpSampling2D(2)(u4)               # 32x32x256
    u3 = scale_attention(u3, [c3, c2, c1])
    u3 = conv_block(u3, 128)                      # 32x32x128

    # Level 2 decoder
    u2 = layers.UpSampling2D(2)(u3)               # 64x64x128
    u2 = scale_attention(u2, [c2, c1])
    u2 = conv_block(u2, 64)                       # 64x64x64

    # Level 1 decoder
    u1 = layers.UpSampling2D(2)(u2)               # 128x128x64
    u1 = scale_attention(u1, [c1])
    u1 = conv_block(u1, 32)                       # 128x128x32

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    return models.Model(inputs, outputs)

# ---------------------------
# 7. Losses & metrics
# ---------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def dice_coeff_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred.flatten() > 0.5).astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# ---------------------------
# 8. Build, compile model
# ---------------------------
model = scale_attention_unetpp()
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
model.summary()

# ---------------------------
# 9. Callbacks
# ---------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint("scale_attention_unetpp_best.h5",
                                                monitor='val_loss',
                                                save_best_only=True,
                                                verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=3, min_lr=1e-6, verbose=1)

# ---------------------------
# 10. Train
# ---------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr]
)

# ---------------------------
# 11. Evaluate (Dice & IoU on validation set)
# ---------------------------
# Predict on val set (smallish, so we predict at once)
preds = model.predict(X_val, batch_size=BATCH_SIZE)

dice_val = dice_coeff_np(y_val, preds)
# IoU
y_flat = y_val.flatten()
p_flat = (preds.flatten() > 0.5).astype(np.float32)
intersection = np.sum(y_flat * p_flat)
union = np.sum(y_flat) + np.sum(p_flat) - intersection
iou_val = (intersection + 1e-6) / (union + 1e-6)

print(f"Dice (val): {dice_val:.4f}")
print(f"IoU  (val): {iou_val:.4f}")

# ---------------------------
# 12. Visualize some predictions
# ---------------------------
n_show = 3
for i in range(n_show):
    idx = np.random.randint(0, len(X_val))
    img = (X_val[idx].squeeze() * 255).astype(np.uint8)
    gt = (y_val[idx].squeeze() * 255).astype(np.uint8)
    pred_mask = (preds[idx].squeeze() > 0.5).astype(np.uint8) * 255

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[pred_mask == 255] = [255, 0, 0]

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Image"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(gt, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Prediction Overlay"); plt.axis('off')
    plt.show()

# ---------------------------
# 13. Save final model
# ---------------------------
model.save("scale_attention_unetpp_final.h5")
print("✅ Model saved as scale_attention_unetpp_final.h5")
