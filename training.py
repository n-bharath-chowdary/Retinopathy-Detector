import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 9
DATASET_DIR = "dataset/train_images"
CSV_PATH = "train.csv"

# Load labels
labels_df = pd.read_csv(CSV_PATH)

# If images are named like 000c1434d8d7.png or .jpeg etc., append extension
labels_df['id_code'] = labels_df['id_code'].astype(str) + '.png'  # change to '.jpeg' or '.jpg' if needed

# Rename columns for ImageDataGenerator
labels_df.columns = ['filename', 'class']
labels_df['class'] = labels_df['class'].astype(str)


# Create ImageDataGenerator for train and validation
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=DATASET_DIR,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=DATASET_DIR,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# CNN Model
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # 5 classes for DR levels
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_retinopathy_model.keras", save_best_only=True, monitor='val_accuracy')

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# Save model
model.save("retinopathy_model.h5")
print("✅ Model training complete and saved.")

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
