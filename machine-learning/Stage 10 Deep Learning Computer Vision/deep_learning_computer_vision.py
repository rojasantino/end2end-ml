#!/usr/bin/env python
# coding: utf-8

# # 🌿 Stage 10: Deep Learning — Computer Vision
# ## Project: Plant Disease Detector
# 

# ## 📦 1. Imports & Configuration

# In[1]:


import os, time, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.model_selection import train_test_split
import cv2

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 96          # use 96×96 for fast CPU training (paper uses 224)
BATCH_SIZE = 32
EPOCHS_CNN = 15
EPOCHS_FT  = 10
N_CLASSES  = 10          # subset of 10 disease classes for fast demo
N_SAMPLES  = 2000        # synthetic images

# ── Plot theme ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0a14', 'axes.facecolor':  '#12121f',
    'axes.edgecolor':   '#3a3a5c', 'text.color':      '#dcdcff',
    'axes.labelcolor':  '#dcdcff', 'xtick.color':     '#9090b8',
    'ytick.color':      '#9090b8', 'grid.color':      '#222240',
    'grid.alpha': 0.5,
})
C1, C2, C3, C4 = '#00d4ff', '#00ff9f', '#ffd700', '#ff6b6b'

print(f"TensorFlow  : {tf.__version__}")
print(f"Image Size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"Classes     : {N_CLASSES}  |  Samples : {N_SAMPLES}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")


# ## 🌱 2. Synthetic PlantVillage Dataset Generation
# 
# We synthesise realistic leaf images using:
# - **Green texture base** (healthy vs diseased colour tones)
# - **Gaussian noise** (sensor noise)
# - **Circular leaf mask** + **vein patterns**
# - **Disease lesions** (dark spots, yellow rings) for diseased classes

# In[ ]:


CLASS_NAMES = [
    'Apple___Apple_scab',       'Apple___Black_rot',
    'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight','Corn___healthy',
    'Tomato___Early_blight',    'Tomato___healthy',
]

# ── Class colour profiles (R,G,B mean) ───────────────────────────────────────
CLASS_PROFILES = [
    (40, 90, 30),    # Apple scab — dark green with spots
    (30, 70, 20),    # Apple black rot — very dark
    (120,110, 40),   # Cedar rust — yellow-orange tinge
    (50, 130, 40),   # Apple healthy — bright green
    (80, 100, 30),   # Corn cercospora — brownish green
    (90,  80, 20),   # Corn rust — orange-rust
    (55, 100, 25),   # Corn blight — pale stripe
    (60, 140, 45),   # Corn healthy — vivid green
    (70,  85, 25),   # Tomato blight — dark spots
    (55, 135, 50),   # Tomato healthy — fresh green
]

def make_leaf_image(class_idx, img_size=96):
    rng  = np.random.default_rng()
    S    = img_size
    rb, gb, bb = CLASS_PROFILES[class_idx]

    # ── Base colour canvas ────────────────────────────────────────────────────
    img  = np.zeros((S, S, 3), dtype=np.float32)
    img[:,:,0] = np.clip(rng.normal(rb, 15, (S,S)), 0, 255)
    img[:,:,1] = np.clip(rng.normal(gb, 20, (S,S)), 0, 255)
    img[:,:,2] = np.clip(rng.normal(bb, 10, (S,S)), 0, 255)

    # ── Circular leaf mask ────────────────────────────────────────────────────
    cx, cy = S//2, S//2
    Y, X   = np.ogrid[:S, :S]
    mask   = ((X-cx)**2/(cx*0.85)**2 + (Y-cy)**2/(cy*0.85)**2) <= 1
    img[~mask] = [20, 30, 10]   # dark background

    # ── Leaf veins ────────────────────────────────────────────────────────────
    for angle in np.linspace(-np.pi/3, np.pi/3, 7):
        x0, y0 = cx, cy
        for t in range(1, S//2):
            xv = int(x0 + t*np.cos(angle))
            yv = int(y0 + t*np.sin(angle))
            if 0 <= xv < S and 0 <= yv < S and mask[yv, xv]:
                img[yv, xv, 1] = np.clip(img[yv, xv, 1]*0.85, 0, 255)

    # ── Disease-specific textures ─────────────────────────────────────────────
    is_diseased = class_idx not in [3, 7, 9]
    if is_diseased:
        n_spots = rng.integers(5, 20)
        for _ in range(n_spots):
            sx = rng.integers(S//4, 3*S//4)
            sy = rng.integers(S//4, 3*S//4)
            if mask[sy, sx]:
                r  = rng.integers(3, 10)
                sy1, sy2 = max(0,sy-r), min(S,sy+r)
                sx1, sx2 = max(0,sx-r), min(S,sx+r)
                spot_mask = mask[sy1:sy2, sx1:sx2]
                # brown/black spot
                img[sy1:sy2, sx1:sx2, 0][spot_mask] *= rng.uniform(1.2, 1.5)
                img[sy1:sy2, sx1:sx2, 1][spot_mask] *= rng.uniform(0.4, 0.6)
                img[sy1:sy2, sx1:sx2, 2][spot_mask] *= rng.uniform(0.3, 0.5)
        # yellow halo for rust classes
        if class_idx in [2, 5]:
            img[:,:,0][mask] = np.clip(img[:,:,0][mask] * 1.3, 0, 255)
            img[:,:,1][mask] = np.clip(img[:,:,1][mask] * 1.1, 0, 255)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

# ── Generate dataset ──────────────────────────────────────────────────────────
print(f"Generating {N_SAMPLES} synthetic leaf images ({IMG_SIZE}×{IMG_SIZE}) …")
t0 = time.time()
n_per_class = N_SAMPLES // N_CLASSES
X_all, y_all = [], []

for cls in range(N_CLASSES):
    for _ in range(n_per_class):
        X_all.append(make_leaf_image(cls, IMG_SIZE))
        y_all.append(cls)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)

# Shuffle
idx   = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]

print(f"Done in {time.time()-t0:.1f}s")
print(f"X shape : {X_all.shape}   dtype: {X_all.dtype}")
print(f"y shape : {y_all.shape}   classes: {np.unique(y_all)}")
print(f"Class balance: {np.bincount(y_all)}")


# ## 🖼️ 3. Sample Leaf Images

# In[ ]:


fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('🌿 Sample Synthetic Leaf Images — 10 Disease Classes',
             fontsize=14, color=C1, fontweight='bold')

for i, ax in enumerate(axes.flat):
    idx = np.where(y_all == i)[0][0]
    ax.imshow(X_all[idx].astype(np.uint8))
    ax.set_title(CLASS_NAMES[i].replace('___',': ').replace('_',' '),
                 fontsize=7.5, color='#dcdcff', pad=4)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_leaves.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Sample grid saved → sample_leaves.png")


# ## 🔍 4. Dataset EDA — Class Distribution & Pixel Statistics
# 

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('📊 Dataset EDA', fontsize=14, color=C1, fontweight='bold')

# ── Class distribution ────────────────────────────────────────────────────────
ax1 = axes[0]
counts  = np.bincount(y_all)
short   = [c.split('___')[1].replace('_',' ')[:18] for c in CLASS_NAMES]
colors  = [C4 if 'healthy' not in c else C2 for c in CLASS_NAMES]
bars    = ax1.barh(short, counts, color=colors, edgecolor='white', linewidth=0.4)
ax1.set_title('Class Distribution', color='#dcdcff')
ax1.set_xlabel('Sample Count')
ax1.axvline(n_per_class, color='white', linewidth=1, linestyle='--', alpha=0.6)
ax1.grid(True, alpha=0.3, axis='x')
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=C4,label='Diseased'), Patch(color=C2,label='Healthy')],
           framealpha=0.3, fontsize=9)

# ── Mean pixel per channel ────────────────────────────────────────────────────
ax2 = axes[1]
mean_r = [X_all[y_all==c][:,:,:,0].mean() for c in range(N_CLASSES)]
mean_g = [X_all[y_all==c][:,:,:,1].mean() for c in range(N_CLASSES)]
mean_b = [X_all[y_all==c][:,:,:,2].mean() for c in range(N_CLASSES)]
x      = np.arange(N_CLASSES)
w      = 0.25
ax2.bar(x-w,  mean_r, w, color='#ff6060', label='Red',   alpha=0.85)
ax2.bar(x,    mean_g, w, color=C2,        label='Green', alpha=0.85)
ax2.bar(x+w,  mean_b, w, color=C1,        label='Blue',  alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(range(N_CLASSES))
ax2.set_title('Mean Pixel Value per Channel', color='#dcdcff')
ax2.set_xlabel('Class'); ax2.set_ylabel('Mean Pixel Value')
ax2.legend(framealpha=0.3); ax2.grid(True, alpha=0.3, axis='y')

# ── Pixel intensity histogram ─────────────────────────────────────────────────
ax3 = axes[2]
sample_img = X_all[:200]
for ch, color, lbl in [(0,'#ff6060','Red'), (1,C2,'Green'), (2,C1,'Blue')]:
    ax3.hist(sample_img[:,:,:,ch].flatten(), bins=60, alpha=0.5,
             color=color, label=lbl, density=True)
ax3.set_title('Pixel Intensity Distribution (200 samples)', color='#dcdcff')
ax3.set_xlabel('Pixel Value'); ax3.legend(framealpha=0.3)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_dataset.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ EDA plot saved → eda_dataset.png")


# ## ⚙️ 5. Preprocessing & Image Augmentation
# 

# In[ ]:


# ── Normalise to [0,1] ────────────────────────────────────────────────────────
X_norm = X_all / 255.0

# ── One-hot encode labels ─────────────────────────────────────────────────────
y_oh   = tf.keras.utils.to_categorical(y_all, N_CLASSES)

# ── Split ─────────────────────────────────────────────────────────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_norm, y_oh,
                                              test_size=0.30, random_state=42,
                                              stratify=y_all)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp,
                                              test_size=0.50, random_state=42,
                                              stratify=y_all[len(X_tr):])

print(f"Train : {X_tr.shape[0]:,} samples")
print(f"Val   : {X_val.shape[0]:,} samples")
print(f"Test  : {X_te.shape[0]:,} samples")

# ── Augmentation pipeline ─────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    horizontal_flip   = True,
    vertical_flip     = True,
    rotation_range    = 25,
    zoom_range        = 0.20,
    width_shift_range = 0.15,
    height_shift_range= 0.15,
    brightness_range  = [0.7, 1.3],
    fill_mode         = 'nearest',
)
datagen.fit(X_tr)

# ── Visualise augmentations ───────────────────────────────────────────────────
sample_leaf = X_tr[0:1]   # shape (1, H, W, 3)
aug_iter    = datagen.flow(sample_leaf, batch_size=1, seed=42)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('⚙️ Image Augmentation — Original + 11 Augmented Versions',
             fontsize=13, color=C1, fontweight='bold')
axes.flat[0].imshow(sample_leaf[0])
axes.flat[0].set_title('Original', color=C2, fontsize=9)
axes.flat[0].axis('off')

for i, ax in enumerate(axes.flat[1:], 1):
    aug = next(aug_iter)[0]
    ax.imshow(np.clip(aug, 0, 1))
    ax.set_title(f'Aug #{i}', color='#9090b8', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('augmentation.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Augmentation grid saved → augmentation.png")


# ## 🧠 6. Custom CNN Architecture
#        

# In[ ]:


def build_custom_cnn(img_size, n_classes):
    inp = keras.Input(shape=(img_size, img_size, 3), name='input')                                                                   

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inp, out, name='PlantCNN')
    return model

cnn_model = build_custom_cnn(IMG_SIZE, N_CLASSES)
cnn_model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy'],
)
cnn_model.summary()
total_params = cnn_model.count_params()
print(f"\nTotal parameters : {total_params:,}")


# ## 🏋️ 7. Train Custom CNN 
# 
# 

# In[ ]:


cnn_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                            monitor='val_accuracy', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.4, patience=3,
                                min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_cnn.keras', save_best_only=True,
                               monitor='val_accuracy', verbose=0),
]

print(f"Training Custom CNN for up to {EPOCHS_CNN} epochs …")
t0 = time.time()

cnn_history = cnn_model.fit(
    datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE, seed=42),
    steps_per_epoch  = len(X_tr) // BATCH_SIZE,
    validation_data  = (X_val, y_val),
    epochs           = EPOCHS_CNN,
    callbacks        = cnn_callbacks,
    verbose          = 1,
)

print(f"\nFinished in {time.time()-t0:.0f}s")
best_val = max(cnn_history.history['val_accuracy'])
print(f"Best Val Accuracy : {best_val:.4f}")


# In[ ]:


def plot_history(history, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, color=C1, fontweight='bold')

    # Accuracy
    axes[0].plot(history['accuracy'],     color=C1,  linewidth=2, label='Train Acc')
    axes[0].plot(history['val_accuracy'], color=C3,  linewidth=2, linestyle='--', label='Val Acc')
    axes[0].set_title('Accuracy', color='#dcdcff')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(framealpha=0.3); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history['loss'],     color=C4,  linewidth=2, label='Train Loss')
    axes[1].plot(history['val_loss'], color=C2,  linewidth=2, linestyle='--', label='Val Loss')
    axes[1].set_title('Loss', color='#dcdcff')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(framealpha=0.3); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight',
                facecolor='#0a0a14', edgecolor='none')
    plt.show()
    print(f"✅ Training curve saved → {fname}")

plot_history(cnn_history.history,
             '📈 Custom CNN — Training History',
             'cnn_training_curves.png')


# ## 🚀 8. MobileNetV2 Transfer Learning

# In[ ]:


def build_mobilenet(img_size, n_classes):
    base = MobileNetV2(
        input_shape = (img_size, img_size, 3),
        include_top = False,
        weights     = None,   # no download needed; random init for demo
    )
    base.trainable = False    # freeze base

    inp = keras.Input(shape=(img_size, img_size, 3))
    x   = preprocess_input(inp)          # MobileNetV2 expects [-1,1]
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inp, out, name='MobileNetV2_PlantDisease')
    return model, base

mob_model, mob_base = build_mobilenet(IMG_SIZE, N_CLASSES)
mob_model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy'],
)
print(f"MobileNetV2 — trainable params: {mob_model.count_params():,}")
print(f"Base frozen: {not mob_base.trainable}")
mob_model.summary(line_length=80)


# ## 🏋️ 9. Train MobileNetV2 — Phase 1 (Head Only)

# In[ ]:


mob_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                            monitor='val_accuracy', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.4, patience=3,
                                min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint('best_mobilenet.keras', save_best_only=True,
                               monitor='val_accuracy', verbose=0),
]

print(f"Training MobileNetV2 head for up to {EPOCHS_CNN} epochs …")
t0 = time.time()

mob_history = mob_model.fit(
    datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE, seed=42),
    steps_per_epoch = len(X_tr) // BATCH_SIZE,
    validation_data = (X_val, y_val),
    epochs          = EPOCHS_CNN,
    callbacks       = mob_callbacks,
    verbose         = 1,
)
print(f"\nPhase 1 done in {time.time()-t0:.0f}s")
print(f"Best Val Accuracy : {max(mob_history.history['val_accuracy']):.4f}")


# ## 🔧 10. Fine-Tuning — Unfreeze Top Layers
# 

# In[ ]:


# Unfreeze last 30 layers of base
mob_base.trainable = True
for layer in mob_base.layers[:-30]:
    layer.trainable = False

trainable_now = sum(1 for l in mob_model.layers if l.trainable)
print(f"Trainable layers after unfreeze: {trainable_now}")

mob_model.compile(
    optimizer = keras.optimizers.Adam(1e-5),   # 100× smaller LR
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy'],
)

print(f"Fine-tuning for up to {EPOCHS_FT} more epochs …")
t0 = time.time()

ft_history = mob_model.fit(
    datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE, seed=42),
    steps_per_epoch = len(X_tr) // BATCH_SIZE,
    validation_data = (X_val, y_val),
    epochs          = EPOCHS_FT,
    callbacks       = mob_callbacks,
    verbose         = 1,
)
print(f"\nFine-tuning done in {time.time()-t0:.0f}s")

# Merge histories for plotting
combined_hist = {}
for k in mob_history.history:
    combined_hist[k] = mob_history.history[k] + ft_history.history[k]

plot_history(combined_hist,
             '📈 MobileNetV2 — Head Training + Fine-Tuning History',
             'mobilenet_training_curves.png')


# ## 📊 11. Model Evaluation on Test Set - 
# 
# > இந்த பகுதியில் Custom CNN மற்றும் MobileNetV2 ஆகிய இரண்டு Models-உம் Test Dataset (300 Images) மீது எப்படி செயல்படுகின்றன என்பதை மதிப்பீடு (Evaluation) செய்கிறோம்.

# In[ ]:


print("Evaluating both models on test set …")

# Custom CNN
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_te, y_te, verbose=0)
cnn_pred_prob = cnn_model.predict(X_te, verbose=0)
cnn_pred      = np.argmax(cnn_pred_prob, axis=1)
y_te_int      = np.argmax(y_te, axis=1)

# MobileNetV2
mob_test_loss, mob_test_acc = mob_model.evaluate(X_te, y_te, verbose=0)
mob_pred_prob = mob_model.predict(X_te, verbose=0)
mob_pred      = np.argmax(mob_pred_prob, axis=1)

print("\n" + "="*60)
print(f"  Custom CNN   — Test Accuracy: {cnn_test_acc:.4f}  Loss: {cnn_test_loss:.4f}")
print(f"  MobileNetV2  — Test Accuracy: {mob_test_acc:.4f}  Loss: {mob_test_loss:.4f}")
print("="*60)

short_names = [c.split('___')[1].replace('_',' ')[:14] for c in CLASS_NAMES]

print("\n── Custom CNN Classification Report ──")
print(classification_report(y_te_int, cnn_pred,
                             target_names=short_names, digits=3))

print("\n── MobileNetV2 Classification Report ──")
print(classification_report(y_te_int, mob_pred,
                             target_names=short_names, digits=3))


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Confusion Matrices — Custom CNN vs MobileNetV2',
             fontsize=13, color=C1, fontweight='bold')

for ax, (pred, title, color) in zip(axes, [
    (cnn_pred, f'Custom CNN  (Acc={cnn_test_acc:.3f})', C1),
    (mob_pred, f'MobileNetV2 (Acc={mob_test_acc:.3f})', C3),
]):
    cm = confusion_matrix(y_te_int, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                cbar=True, linewidths=0.3, linecolor='#0a0a14',
                xticklabels=[s[:8] for s in short_names],
                yticklabels=[s[:8] for s in short_names])
    ax.set_title(title, color=color, fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7.5)
    plt.setp(ax.get_yticklabels(), fontsize=7.5)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Confusion matrices saved → confusion_matrices.png")


# ## 🔥 12. Grad-CAM — Gradient-weighted Class Activation Maps
# 

# In[ ]:


def make_gradcam(model, img_array, last_conv_name, class_idx=None):
    grad_model = Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array[np.newaxis], training=False)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads     = tape.gradient(loss, conv_out)
    pooled_g  = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out  = conv_out[0]
    cam       = tf.reduce_sum(tf.multiply(pooled_g, conv_out), axis=-1)
    cam       = np.maximum(cam.numpy(), 0)
    cam       = cam / (cam.max() + 1e-8)
    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    return cam_resized, int(class_idx)

# Find last conv layer in custom CNN
last_conv = [l.name for l in cnn_model.layers if 'conv' in l.name][-1]
print(f"Last conv layer: {last_conv}")

# Pick one sample per class
fig, axes = plt.subplots(3, N_CLASSES, figsize=(22, 7))
fig.suptitle('🔥 Grad-CAM — What the CNN Sees for Each Disease Class',
             fontsize=13, color=C4, fontweight='bold')

for cls in range(N_CLASSES):
    idx = np.where(y_te_int == cls)[0]
    if len(idx) == 0:
        continue
    img  = X_te[idx[0]]              # normalised [0,1]
    cam, pred_cls = make_gradcam(cnn_model, img, last_conv, cls)

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img + 0.5 * heatmap

    axes[0, cls].imshow(img)
    axes[0, cls].set_title(short_names[cls], fontsize=7, color='#dcdcff')
    axes[0, cls].axis('off')

    axes[1, cls].imshow(cam, cmap='jet')
    axes[1, cls].set_title('CAM', fontsize=7, color=C4)
    axes[1, cls].axis('off')

    axes[2, cls].imshow(np.clip(overlay, 0, 1))
    correct = '+' if pred_cls == cls else 'x'
    axes[2, cls].set_title(f'{correct} {short_names[pred_cls][:8]}',
                            fontsize=7, color=C2 if pred_cls==cls else C4)
    axes[2, cls].axis('off')

axes[0,0].set_ylabel('Original', color='#dcdcff', fontsize=9)
axes[1,0].set_ylabel('Grad-CAM',  color='#dcdcff', fontsize=9)
axes[2,0].set_ylabel('Overlay',   color='#dcdcff', fontsize=9)

plt.tight_layout()
plt.savefig('gradcam.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Grad-CAM plot saved → gradcam.png")


# ## 🏆 13. Model Comparison Dashboard
# 

# In[ ]:


fig = plt.figure(figsize=(18, 10))
fig.suptitle('🏆 Plant Disease Detector — Model Comparison Dashboard',
             fontsize=16, color=C3, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# Training curves overlay
ax1 = fig.add_subplot(gs[0, 0:2])
ep_cnn = range(1, len(cnn_history.history['val_accuracy'])+1)
ep_mob = range(1, len(combined_hist['val_accuracy'])+1)
ax1.plot(ep_cnn, cnn_history.history['val_accuracy'], color=C1,
         linewidth=2, label='CNN Val Acc', marker='o', ms=4)
ax1.plot(ep_mob, combined_hist['val_accuracy'], color=C3,
         linewidth=2, label='MobileNetV2 Val Acc', marker='s', ms=4)
ft_start = len(mob_history.history['val_accuracy'])
ax1.axvline(ft_start, color=C4, linewidth=1.2, linestyle='--', alpha=0.7)
ax1.text(ft_start+0.2, 0.4, 'Fine-tune start', color=C4, fontsize=9)
ax1.set_title('Validation Accuracy — Training Progress', color='#dcdcff')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val Accuracy')
ax1.legend(framealpha=0.3); ax1.grid(True, alpha=0.3)

# Per-class accuracy
ax2 = fig.add_subplot(gs[0, 2])
cnn_per_class = [
    accuracy_score(y_te_int[y_te_int==c], cnn_pred[y_te_int==c])
    if (y_te_int==c).sum() > 0 else 0
    for c in range(N_CLASSES)
]
mob_per_class = [
    accuracy_score(y_te_int[y_te_int==c], mob_pred[y_te_int==c])
    if (y_te_int==c).sum() > 0 else 0
    for c in range(N_CLASSES)
]
x = np.arange(N_CLASSES); w = 0.38
ax2.bar(x-w/2, cnn_per_class, w, color=C1, alpha=0.85, label='CNN', edgecolor='white', lw=0.4)
ax2.bar(x+w/2, mob_per_class, w, color=C3, alpha=0.85, label='MobileNetV2', edgecolor='white', lw=0.4)
ax2.set_xticks(x); ax2.set_xticklabels(range(N_CLASSES))
ax2.set_title('Per-Class Accuracy', color='#dcdcff')
ax2.set_xlabel('Class'); ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1.1); ax2.legend(framealpha=0.3, fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Overall metrics bar
ax3 = fig.add_subplot(gs[1, 0])
model_labels = ['CNN', 'MobileNetV2']
test_accs    = [cnn_test_acc, mob_test_acc]
bars = ax3.bar(model_labels, test_accs, color=[C1, C3],
               edgecolor='white', linewidth=0.5, alpha=0.85)
for b, v in zip(bars, test_accs):
    ax3.text(b.get_x()+b.get_width()/2, v+0.01,
             f'{v:.4f}', ha='center', fontsize=12, color='white', fontweight='bold')
ax3.set_title('Test Accuracy', color='#dcdcff')
ax3.set_ylim(0, 1.15); ax3.grid(True, alpha=0.3, axis='y')

# Confidence histogram
ax4 = fig.add_subplot(gs[1, 1])
cnn_conf = cnn_pred_prob.max(axis=1)
mob_conf = mob_pred_prob.max(axis=1)
ax4.hist(cnn_conf, bins=25, color=C1, alpha=0.65, label='CNN',         density=True)
ax4.hist(mob_conf, bins=25, color=C3, alpha=0.65, label='MobileNetV2', density=True)
ax4.set_title('Prediction Confidence Distribution', color='#dcdcff')
ax4.set_xlabel('Max Softmax Probability')
ax4.legend(framealpha=0.3); ax4.grid(True, alpha=0.3)

# Summary table
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
tbl_data = [
    ['Custom CNN',   f'{cnn_test_acc:.4f}', f'{cnn_test_loss:.4f}', f'{cnn_model.count_params():,}'],
    ['MobileNetV2',  f'{mob_test_acc:.4f}', f'{mob_test_loss:.4f}', f'{mob_model.count_params():,}'],
]
tbl = ax5.table(
    cellText  = tbl_data,
    colLabels = ['Model','Accuracy','Loss','Params'],
    cellLoc   = 'center', loc='center',
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1a30' if r > 0 else '#2a2a5a')
    cell.set_edgecolor('#4a4a7a')
    cell.set_text_props(color='white')
ax5.set_title('Model Summary', color='#dcdcff', pad=10)

plt.savefig('comparison_dashboard.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Dashboard saved → comparison_dashboard.png")


# ## 🔎 14. Live Prediction Demo — Single Image Inference

# In[ ]:


def predict_image(model, img_array, class_names, top_k=3):
    probs    = model.predict(img_array[np.newaxis], verbose=0)[0]
    top_idx  = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

sample_indices = np.random.choice(len(X_te), 5, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('🔎 Prediction Demo — Top-3 Class Probabilities (MobileNetV2)',
             fontsize=13, color=C3, fontweight='bold')

for col, idx in enumerate(sample_indices):
    img   = X_te[idx]
    true  = y_te_int[idx]
    preds = predict_image(mob_model, img, CLASS_NAMES)
    pred_cls_idx = CLASS_NAMES.index(preds[0][0])

    ax_img = axes[0, col]
    ax_img.imshow(img)
    is_correct = pred_cls_idx == true
    ax_img.set_title(
        f"True: {short_names[true]}",
        color=C2 if is_correct else C4, fontsize=8, fontweight='bold')
    ax_img.axis('off')

    ax_bar = axes[1, col]
    names  = [p[0].split('___')[1].replace('_',' ')[:14] for p in preds]
    vals   = [p[1] for p in preds]
    bar_colors = [C2 if CLASS_NAMES.index(p[0])==true else C1 for p in preds]
    ax_bar.barh(names[::-1], vals[::-1], color=bar_colors[::-1],
                edgecolor='white', linewidth=0.4)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel('Probability', fontsize=8)
    ax_bar.tick_params(labelsize=7.5)
    ax_bar.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('prediction_demo.png', dpi=120, bbox_inches='tight',
            facecolor='#0a0a14', edgecolor='none')
plt.show()
print("✅ Prediction demo saved → prediction_demo.png")


# ## 🌐 15. Streamlit Web App — Farmer Upload Portal
# 
# The cell below writes `app.py` to disk.  
# Run with: `streamlit run app.py`
# 

# In[ ]:


app_src_lines = ['import streamlit as st', 'import tensorflow as tf', 'import numpy as np', 'import cv2', 'from PIL import Image', '', "st.set_page_config(page_title='Plant Disease Detector', page_icon='🌿', layout='wide')", '', 'CLASS_NAMES = [', "    'Apple___Apple_scab', 'Apple___Black_rot',", "    'Apple___Cedar_apple_rust', 'Apple___healthy',", "    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust',", "    'Corn___Northern_Leaf_Blight', 'Corn___healthy',", "    'Tomato___Early_blight', 'Tomato___healthy',", ']', '', 'TREATMENT = {', "    'Apple___Apple_scab'           : 'Apply fungicide (captan/mancozeb). Remove infected leaves.',", "    'Apple___Black_rot'            : 'Prune infected branches. Apply copper-based fungicide.',", "    'Apple___Cedar_apple_rust'     : 'Remove nearby cedar trees. Apply myclobutanil fungicide.',", "    'Apple___healthy'              : 'No disease detected. Continue regular care.',", "    'Corn___Cercospora_leaf_spot'  : 'Apply strobilurin fungicide. Improve air circulation.',", "    'Corn___Common_rust'           : 'Apply fungicide early. Use rust-resistant varieties.',", "    'Corn___Northern_Leaf_Blight'  : 'Apply propiconazole fungicide. Rotate crops.',", "    'Corn___healthy'               : 'No disease detected. Continue regular care.',", "    'Tomato___Early_blight'        : 'Remove affected leaves. Apply chlorothalonil fungicide.',", "    'Tomato___healthy'             : 'No disease detected. Continue regular care.',", '}', '', 'IMG_SIZE = 96', '', '@st.cache_resource', 'def load_model():', "    return tf.keras.models.load_model('best_mobilenet.keras')", '', 'def preprocess(img_pil, size=IMG_SIZE):', "    img = np.array(img_pil.convert('RGB').resize((size, size)))", '    return img.astype(np.float32) / 255.0', '', "st.title('🌿 Plant Disease Detector')", "st.markdown('**AI-powered leaf disease classification for farmers**')", '', 'col1, col2 = st.columns([1, 1])', '', 'with col1:', "    st.subheader('📷 Upload a Leaf Image')", "    uploaded = st.file_uploader('Choose a leaf photo', type=['jpg','jpeg','png'])", '    if uploaded:', '        img_pil = Image.open(uploaded)', "        st.image(img_pil, caption='Uploaded Leaf', use_column_width=True)", '', 'with col2:', '    if uploaded:', "        st.subheader('🔍 Diagnosis')", "        with st.spinner('Analysing ...'):", '            try:', '                model = load_model()', '                img   = preprocess(img_pil)', '                probs = model.predict(img[np.newaxis], verbose=0)[0]', '                top3  = np.argsort(probs)[::-1][:3]', '                pred_class = CLASS_NAMES[top3[0]]', '                pred_conf  = probs[top3[0]]', "                is_healthy = 'healthy' in pred_class.lower()", "                badge = '✅ Healthy' if is_healthy else '⚠️ Disease Detected'", "                st.markdown(f'### {badge}')", '                st.markdown(f"**Predicted:** `{pred_class.replace(\'___\',\': \').replace(\'_\',\' \')}`")', "                st.metric('Confidence', f'{pred_conf:.1%}')", "                st.markdown('---')", "                st.markdown('### 💊 Treatment')", "                st.info(TREATMENT.get(pred_class, 'Consult an agronomist.'))", "                st.markdown('### 📊 Top-3 Predictions')", '                for i, idx in enumerate(top3):', "                    name = CLASS_NAMES[idx].replace('___',': ').replace('_',' ')", "                    st.progress(float(probs[idx]), text=f'{i+1}. {name} — {probs[idx]:.1%}')", '            except Exception as e:', "                st.error(f'Model not found. Train model first.\\n{e}')", '    else:', "        st.info('👆 Upload a leaf image to get started')", '', "st.markdown('---')", "st.caption('Stage 10 Portfolio — MobileNetV2 | TensorFlow + Streamlit')"]
app_src = '\n'.join(app_src_lines)
with open('app.py', 'w') as _f:
    _f.write(app_src)
print('✅ Streamlit app written → app.py')
print('Run: streamlit run app.py')


# 
# ### GitHub README Excerpt
# > Built a plant disease classifier on synthetic PlantVillage-style data (10 disease classes). Trained a custom 4-block CNN and MobileNetV2 with transfer learning + fine-tuning. Implemented Grad-CAM for visual explainability. Deployed as a Streamlit web app where farmers upload leaf photos and receive disease diagnosis with treatment recommendations.
# 
