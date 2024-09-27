import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
from tensorflow.keras.callbacks import ModelCheckpoint

from model_script_5 import Model_CNN as Model_5
from model_script_5_2 import Model_CNN as Model_5_2
from model_script_6 import Model_CNN as Model_6
from model_script_7 import Model_CNN as Model_7
from model_script_8 import Model_CNN as Model_8
from model_script_9 import Model_CNN as Model_9
from model_script_10 import Model_CNN as Model_10


#change based on which model(s) you'd like to train
MODELS = [
     Model_5,
]

# Directories
TRAINED_MODELS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2'
DATASET_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\dataset_2\processed'
SAVE_PLOTS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\metrics_plots\metrics_plots_2'

# ImageDataGenerator for data augmentation
# Will randomly flip, shift, rotate images
img_datagen = IMG(
    rescale = 1./255, # normalise to values between 0 and 1 to feed into model
    validation_split = 0.2,  # 80% for training, 20% for validation
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True  
)

# Image dataset conversion into a dataframe of two columns
def img_dataframe(DATASET_DIR):
    file_paths = []
    labels = []

    # Assign labels to folders
    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.isdir(folder_path):
            if folder_name == 'b_me':
                label = '1'
            elif folder_name == 'a_not_me':
                label = '0'
            else:
                continue

        # Append filenames and labels to empty lists created earlier
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)
            labels.append(label)

    # Create dataframe
    img_df = pd.DataFrame({
        'filename': file_paths,
        'label': labels
    })

    return img_df


def train_model(model_name, model_fn, img_df):
    training_datagen = img_datagen.flow_from_dataframe(
        img_df,
        x_col='filename',
        y_col='label',
        target_size=(75,75),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_datagen = img_datagen.flow_from_dataframe(
        img_df,
        x_col='filename',
        y_col='label',
        target_size=(75,75),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Call model function to get model architecture
    model = model_fn(input_shape=(75,75,1))

    # Create path to save models from each fold
    model_path = os.path.join(TRAINED_MODELS_DIR, f'{model_name}.keras')
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        save_best_only=True,
        mode='auto',
        verbose=1
    )

    # Train the model
    history = model.fit(
        x=training_datagen,
        validation_data=validation_datagen,
        epochs=10,
        callbacks=[checkpoint]
    )

    return history


def plot_save_metrics(history, model_name):
    model_plot_dir = os.path.join(SAVE_PLOTS_DIR, model_name)

    if not os.path.exists(model_plot_dir):
        os.makedirs(model_plot_dir)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_plot_dir, f'{model_name}_accuracy_plot.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_plot_dir, f'{model_name}_loss_plot.png'))
    plt.close()


def main():
    # Load image dataframe
    img_df = img_dataframe(DATASET_DIR)

    # Loop over all models
    for i, model_i in enumerate(MODELS, start = 6):  # Starting index from 7
        model_name = f'model_{i}'

        print(f"Training {model_name}...")

        # Train the model
        history = train_model(model_name, model_i, img_df)

        # Plot and save metrics
        plot_save_metrics(history, model_name)

        print(f"{model_name} training complete and plots saved.")


if __name__ == '__main__':
    main()
