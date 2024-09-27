import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Layers will stack in sequential order...

def Model_CNN(input_shape = (75,75,1)):
    model = Sequential()

    #Layer 1(Conv + LeakyReLU + Pool)
    model.add(Conv2D(filters = 256, kernel_size = (3,3), input_shape = (75,75,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Layer 2(Conv + LeakyReLU + Pool)      
    model.add(Conv2D(filters = 128, kernel_size = (3,3), input_shape = (75,75,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Layer 3(Conv + LeakyReLU + Pool)          
    model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = (75,75,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Layer 4(Conv + LeakyReLU + Pool)
    model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (75,75,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Flatten layer       
    model.add(Flatten())

    #Fully Connected Layer 1
    model.add(Dense(units = 128))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dropout(0.5))

    #Fully Connected Layer 2. Sigmoid for binary, Softmax for multi class classification
    model.add(Dense(units = 1 , activation = 'sigmoid'))

    #Model Compilation
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

model = Model_CNN()
model.summary()
