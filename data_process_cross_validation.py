import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint

#import my model from model script
from model_script import Model_CNN #*********************************************************REMEMBER TO CHANGE!!!!!!!

# Directories
TRAINED_MODELS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2'
DATASET_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\dataset_2\processed'
SAVE_PLOTS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\metrics_plots\metrics_plots_2'

# ImageDataGenerator with data augmentation 
# also splits data into train val sets, while randomly flipping tilting etc images
img_datagen = IMG(
    rescale=1./255, #set between 0 and 1 for input into model training
    validation_split = 0.2,  # 80% for training, 20% for validation
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    vertical_flip=True  
    )

# Image dataset conversion into a dataframe of two columns
# Column 1 = filenames, Column 2 = labels

def img_dataframe(DATASET_DIR):
    file_paths = []
    labels = []

    #assign labels to folders
    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.isdir(folder_path):
            if folder_name == 'b_me':
                label = '1'
            elif folder_name == 'a_not_me':
                label = '0'
            else:
                continue    

        #append filenames and labels to empty lists created earleir
        for file_name in os.listdir(folder_path):
             file_path = os.path.join(folder_path, file_name)
             file_paths.append(file_path)
             labels.append(label)

    #create dataframe
    img_df = pd.DataFrame({
         'filename' : file_paths,
         'label' : labels
    })

    return img_df


# ****Now must create function to perform k fold cross validation****


def k_fold_cross_validation(k=5, img_df = None):
     results = []
     file_names = img_df['filename']
     labels = img_df['label']
     skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)

     #for i, (train_index, test_index) in enumerate(skf.split(X, y)):
     for fold_index, (train_index, test_index) in enumerate(skf.split(file_names,labels)):
          
        #create dataframes for each fold iteration
        train_df = pd.DataFrame({
             'filename': file_names.iloc[train_index].values,
             'label': labels.iloc[train_index].astype(str).values
        })

        val_df = pd.DataFrame({
             'filename': file_names.iloc[test_index].values,
             'label' : labels.iloc[test_index].astype(str).values
        })

        #flow from dataframe
        training_datagen = img_datagen.flow_from_dataframe(
            train_df,
            x_col = 'filename',
            y_col = 'label',
            target_size=(75,75),
            color_mode='grayscale',
            batch_size = 32,
            class_mode='binary',
            shuffle = True
        )

        validation_datagen = img_datagen.flow_from_dataframe(
             val_df,
             x_col = 'filename',
             y_col = 'label',
             target_size = (75,75),
             color_mode = 'grayscale',
             batch_size = 32,
             class_mode = 'binary',
             shuffle = False
        )

        model = Model_CNN(input_shape = (75,75,1))

        #create path to save models from each fold
        model_path = os.path.join(TRAINED_MODELS_DIR, f'model_1_{fold_index + 1}.keras')#/////////////////////////////////////////////////////////REMEMBER TO CHANGE!!
        checkpoint = ModelCheckpoint(
            filepath = model_path,
            save_best_only = True,
            mode = 'auto',
            verbose = 1
        )        

        history = model.fit(
             x = training_datagen,
             validation_data = validation_datagen,
             epochs = 10,
             callbacks = [checkpoint]
        )
        # Get results
        results.append({
            'fold': fold_index + 1,
            'history': history.history
        })
     return results


# Function to plot and save metrics per fold
def metrics(metric_name, results, title, ylabel):
    for i, result in enumerate(results):
        plt.figure(figsize=(12, 8))
        plt.plot(result['history'][metric_name], label=f'Fold {i+1}')
        plt.title(f"{title} - Fold {i+1}")
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(SAVE_PLOTS_DIR,f"{metric_name}_fold_{i+1}_plot.png"))
        plt.close()
        print (f'Saved{metric_name}plot for fold {i+1} to {SAVE_PLOTS_DIR}')

def main():
  
    # Run K-Fold Cross Validation
    img_df = img_dataframe(DATASET_DIR)
    results = k_fold_cross_validation(k = 5, img_df = img_df)

    # Now create and save plots for training/validation accuracy, loss vs epochs 

    if not os.path.exists(SAVE_PLOTS_DIR):
        os.makedirs(SAVE_PLOTS_DIR)

    # Plot and save metrics
    metrics('accuracy', results, 'Training Accuracy vs Epochs', 'Accuracy')
    metrics('val_accuracy', results, 'Validation Accuracy vs Epochs', 'Validation Accuracy')
    metrics('loss', results, 'Training Loss vs Epochs', 'Loss')
    metrics('val_loss', results, 'Validation Loss vs Epochs', 'Validation Loss')

    # Print summary results for each fold
    for result in results:
        print(f"Fold {result['fold']} - Final val_accuracy: {result['history']['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()