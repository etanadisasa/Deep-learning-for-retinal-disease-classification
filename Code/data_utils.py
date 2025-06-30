# Etana Disasa \\
# Department of Data Science \\
# College of Computer and Information Sciences \\
# Regis University \\
# 3333 Regis Boulevard, Denver, CO 80221 \\
#----------------------------------------------------
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === Paths ===
BASE_DIR = 'Dataset'
GT_DIR = os.path.join(BASE_DIR, 'Groundtruths')
IMG_ROOT = os.path.join(BASE_DIR, 'OriginalImages')

# === CSV Paths ===
RFMID_TRAIN_CSV = os.path.join(GT_DIR, 'RFMiDTrainingLabels.csv')
RFMID_VAL_CSV = os.path.join(GT_DIR, 'RFMiDValidationLabels.csv')
APTOS_TRAIN_CSV = os.path.join(GT_DIR, 'APTOSTrainingLabels.csv')
APTOS_VAL_CSV = os.path.join(GT_DIR, 'APTOSValidationLabels.csv')

# === Loaders for individual datasets ===

def load_rfmid_with_paths():
    rfmid_train = pd.read_csv(RFMID_TRAIN_CSV)
    rfmid_val = pd.read_csv(RFMID_VAL_CSV)
    rfmid_train['filename'] = 'RFMiDTrainingSet/' + rfmid_train['ID'].astype(str) + '.png'
    rfmid_val['filename'] = 'RFMiDValidationSet/' + rfmid_val['ID'].astype(str) + '.png'
    rfmid_train = rfmid_train.rename(columns={'DR': 'label'})
    rfmid_val = rfmid_val.rename(columns={'DR': 'label'})
    rfmid_train['label'] = rfmid_train['label'].astype(str)
    rfmid_val['label'] = rfmid_val['label'].astype(str)
    return rfmid_train, rfmid_val, IMG_ROOT, IMG_ROOT

def load_aptos_with_paths():
    aptos_train = pd.read_csv(APTOS_TRAIN_CSV)
    aptos_val = pd.read_csv(APTOS_VAL_CSV)
    aptos_train['filename'] = 'APTOSTrainingSet/' + aptos_train['id_code'].astype(str) + '.png'
    aptos_val['filename'] = 'APTOSValidationSet/' + aptos_val['id_code'].astype(str) + '.png'
    aptos_train = aptos_train.rename(columns={'binary_label': 'label'})
    aptos_val = aptos_val.rename(columns={'binary_label': 'label'})
    aptos_train['label'] = aptos_train['label'].astype(str)
    aptos_val['label'] = aptos_val['label'].astype(str)
    return aptos_train, aptos_val, IMG_ROOT, IMG_ROOT

def load_combined_with_paths():
    rfmid_train, rfmid_val, _, _ = load_rfmid_with_paths()
    aptos_train, aptos_val, _, _ = load_aptos_with_paths()
    combined_train = pd.concat([rfmid_train, aptos_train], ignore_index=True)
    combined_val = pd.concat([rfmid_val, aptos_val], ignore_index=True)
    return combined_train, combined_val, IMG_ROOT, IMG_ROOT

# === Data Generator ===

def get_generators(dataset='combined', batch_size=32, target_size=(224,224)):
    print(f"👉 Dataset requested: {dataset}")
    if dataset == 'rfmid':
        train_df, val_df, train_dir, val_dir = load_rfmid_with_paths()
    elif dataset == 'aptos':
        train_df, val_df, train_dir, val_dir = load_aptos_with_paths()
    elif dataset == 'combined':
        train_df, val_df, train_dir, val_dir = load_combined_with_paths()
    else:
        raise ValueError("Dataset must be one of: 'rfmid', 'aptos', or 'combined'.")

    y_col = 'label'
    class_mode = 'binary'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='filename',
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_dir,
        x_col='filename',
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    return train_generator, val_generator

# === Model Builder ===

def build_finetuned_model(num_classes=1, input_shape=(224,224,3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    if num_classes == 1:
        predictions = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])
    return model
