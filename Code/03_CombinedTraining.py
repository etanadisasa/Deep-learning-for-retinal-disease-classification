import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import your data_utils functions (make sure data_utils.py is in the same folder or in PYTHONPATH)
from data_utils import get_generators, build_finetuned_model

# === Parameters ===
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 20

# Create directories if they don't exist
os.makedirs('Models', exist_ok=True)
os.makedirs('History', exist_ok=True)

# No need to declare the dataset source as this combines both sources. 

# === Load Combined Data ===
train_gen, val_gen = get_generators(dataset='combined', batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

# === Build Model ===
model = build_finetuned_model(num_classes=1, input_shape=TARGET_SIZE + (3,))

# === Callbacks ===
checkpoint_filepath = 'Models/combined_best_model.keras'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)
]

print("Combined Train samples:", train_gen.samples)
print("Combined Validation samples:", val_gen.samples)

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# === Train the model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# === Plot training history ===
def plot_history(history):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy
    axs[0].plot(history.history.get('accuracy', []), label='Train Accuracy', marker='o')
    axs[0].plot(history.history.get('val_accuracy', []), label='Validation Accuracy', marker='s')
    axs[0].set_title('Accuracy over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    
    # Loss
    axs[1].plot(history.history.get('loss', []), label='Train Loss', marker='o')
    axs[1].plot(history.history.get('val_loss', []), label='Validation Loss', marker='s')
    axs[1].set_title('Loss over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# === Save the best model in .h5 format ===
best_model = load_model(checkpoint_filepath, compile=False)
best_model.save('Models/combined_best_model.h5', save_format='h5')

print("✅ Model saved as 'Models/combined_best_model.h5'")

# === Save training history for later use ===
history_path = 'History/combined_history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

print(f"✅ Training history saved as '{history_path}'")
