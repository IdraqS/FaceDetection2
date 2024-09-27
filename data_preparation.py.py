import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
import os
import numpy as np
import matplotlib.pyplot as plt

# Dataset directory path
DATASET_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\dataset\processed'

# Image Data augmentation
# Splits into val and train sets then random flips, rotates, shifts etc
img_datagen = IMG(
    rescale = 1./255,
    validation_split = 0.2,  # 80% for training, 20% for validation
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Training data generator
def TrainingDataGenerator(DATASET_DIR, img_datagen):
    training_datagen = img_datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (75, 75),
        color_mode = 'grayscale',
        batch_size = 32,
        class_mode = 'binary',
        subset = 'training' 
    )
    return training_datagen

# Validation data generator
def ValidationDataGenerator(DATASET_DIR, img_datagen):
    validation_datagen = img_datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (75, 75),
        color_mode = 'grayscale',
        batch_size = 32,
        class_mode = 'binary',
        subset = 'validation'  # Using 'validation' subset
    )
    return validation_datagen

# Generate training and validation datasets
train_gen = TrainingDataGenerator(DATASET_DIR, img_datagen)
val_gen = ValidationDataGenerator(DATASET_DIR, img_datagen)

# Show random pics from generator to show if assigns 1 and 0 Correctly
plt.figure(figsize = (8,8))
for images, labels in train_gen: 
    for i in range(9): 
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((images[i] * 255).astype("uint8").squeeze(), cmap='gray') #multiply by 255 to normalise pixel values
        plt.title(int(labels[i]))  
        plt.axis("off")
    break 
plt.show()

