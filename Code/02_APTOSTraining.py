import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import your utility functions
from data_utils import get_generators, build_finetuned_model

# === Parameters ===
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 20

# === Set dataset name ===
DATASET_NAME = 'aptos'  # or 'rfmid' this pretty much decides which dataset to consider. 

# === Get generators ===
train_gen, val_gen = get_generators(dataset=DATASET_NAME, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

# === Build model ===
# === Build model ===
if DATASET_NAME in ['rfmid', 'aptos']:
    num_classes = 1
else:
    # default or raise error
    num_classes = 1 # Adjust based on your dataset classes. I wanted to keep it like this for future changes. 
model = build_finetuned_model(num_classes=num_classes, input_shape=TARGET_SIZE + (3,))

# === Prepare directories ===
os.makedirs('Models', exist_ok=True)
os.makedirs('History', exist_ok=True)

checkpoint_filepath = f'Models/{DATASET_NAME}_best_model.keras'
history_path = f'History/{DATASET_NAME}_history.pkl'

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)
]

print(f"{DATASET_NAME.upper()} Train samples:", train_gen.samples)
print(f"{DATASET_NAME.upper()} Validation samples:", val_gen.samples)

# === Set seeds for reproducibility ===
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# === Train model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save training history ===
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ Training history saved to {history_path}")

# === Plot training history ===
def plot_history(history):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].plot(history['accuracy'], label='Train Accuracy')
    axs[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[0].legend()
    axs[0].set_title('Accuracy')
    axs[1].plot(history['loss'], label='Train Loss')
    axs[1].plot(history['val_loss'], label='Validation Loss')
    axs[1].legend()
    axs[1].set_title('Loss')
    plt.show()

plot_history(history.history)

# === Save model in .h5 format for compatibility ===
best_model = load_model(checkpoint_filepath, compile=False)
best_model.save(f'Models/{DATASET_NAME}_best_model.h5', save_format='h5')

print(f"✅ {DATASET_NAME.upper()} model training complete and saved.")
