"""
Diabetic Retinopathy Detection using EfficientNet-B0
Author: [Your Name]
Project: Final Year Project - Alliance University
Description: 
    This script trains a Deep Learning model to detect 5 stages of Diabetic Retinopathy.
    It uses Transfer Learning with EfficientNet-B0, Class Weighting for imbalance,
    and a Two-Phase training strategy (Head Training -> Fine Tuning).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Adjust these paths to match your local machine
BASE_PATH = "/Users/aditya/Desktop/dr_project/aptos2019-blindness-detection/"
CSV_PATH = os.path.join(BASE_PATH, "train.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "train_images")

# Hyperparameters
IMG_SIZE = 224          # EfficientNet-B0 Standard
NUM_CLASSES = 5         # 0:No DR, 1:Mild, 2:Mod, 3:Severe, 4:Proliferative
BATCH_SIZE = 32         # Good balance for CPU training
EPOCHS_HEAD = 10        # Phase 1: Train classifier only
EPOCHS_FINE = 15        # Phase 2: Fine-tune top layers

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print(f"[INFO] Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Format dataframe for flow_from_dataframe
df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png" if not str(x).endswith('.png') else x)
df['diagnosis'] = df['diagnosis'].astype(str)

print(f"[INFO] Found {len(df)} images.")

# Data Augmentation (Prevents Overfitting)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,       # 80% Train, 20% Validation
    rotation_range=20,          # Rotate +/- 20 degrees
    zoom_range=0.2,             # Zoom in/out 20%
    horizontal_flip=True,       # Flip Left/Right
    vertical_flip=True,         # Flip Up/Down (Valid for eyes)
    fill_mode='constant'
)

# Generators
train_gen = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False 
)

# Calculate Class Weights (Handles Dataset Imbalance)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"[INFO] Class Weights: {class_weights_dict}")

# ==========================================
# 3. MODEL ARCHITECTURE (EfficientNet-B0)
# ==========================================
def build_model():
    # Load Pre-trained ImageNet weights (exclude top layers)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Custom Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)                 # Strong dropout to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

model, base_model = build_model()

# Define Callbacks
checkpoint = ModelCheckpoint('best_dr_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', verbose=1)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.2, monitor='val_loss', verbose=1)

# ==========================================
# 4. PHASE 1: TRAIN HEAD (FROZEN BASE)
# ==========================================
print("\n[INFO] Phase 1: Training Classification Head (Base Frozen)...")
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==========================================
# 5. PHASE 2: FINE-TUNING (UNFREEZE TOP LAYERS)
# ==========================================
print("\n[INFO] Phase 2: Fine-Tuning Top 20 Layers...")
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Low LR to prevent wrecking weights
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================
print("\n[INFO] Evaluating Final Model...")
val_gen.reset()
predictions = model.predict(val_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No DR', 'Mild', 'Mod', 'Severe', 'Proliferative']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif'], yticklabels=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Diabetic Retinopathy Detection')
plt.savefig('confusion_matrix.png')
print("[INFO] Confusion matrix saved as 'confusion_matrix.png'")

# Save Final Model
model.save("final_dr_efficientnet_b0.keras")
print("[INFO] Model saved successfully.")
