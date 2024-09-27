import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_process_cross_validation import img_dataframe

# Directories
TEST_DATASET_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\testing_dataset\processed'
TRAINED_MODELS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models_2'
SAVE_TEST_PLOTS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\metrics_plots'
SAVE_RESULTS_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\excel_sheets'


def main():
    results = []
    test_datagen = IMG(rescale=1./255)
    test_df = img_dataframe(TEST_DATASET_DIR)

    test_gen = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        x_col = 'filename',
        y_col = 'label',
        target_size = (75,75),
        color_mode = 'grayscale',
        batch_size = 32,
        class_mode = 'binary',
        shuffle = False #no shuffling test set!
    )

    models = [
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_1_1.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_1_2.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_1_3.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_1_4.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_1_5.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_2_1.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_2_2.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_2_3.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_2_4.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_2_5.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_6.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_7.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_8.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_9.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_10.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_11_1.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_11_2.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_11_3.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_11_4.keras',
        r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_2\model_11_5.keras'
    ]
    
    for i, model_path in enumerate(models):
        model = tf.keras.models.load_model(model_path)
        test_gen.reset()

        # generate binary prediction. Create 0.5 threshold
        y_pred = model.predict(x=test_gen, steps=len(test_gen))
        y_pred_binary = (y_pred > 0.5).astype(int) #if pred > 0.5 == 1, if pred < 0.1 == 0
        y_true = test_gen.classes


        #calculate metrics 
        accuracy = accuracy_score(y_true, y_pred_binary)    
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)

        # Print metrics for current iteration
        print(f"\n{os.path.basename(model_path)}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

        #append to empty results list
        results.append({
            'model': f'Model {os.path.basename(model_path)}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        plt.bar([r['model'] for r in results], [r[metric] for r in results])
        plt.title(f'{metric.capitalize()} for each model')
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_TEST_PLOTS_DIR, f'{metric}_plot.png'))
        plt.close()


if __name__ == '__main__':
    main()