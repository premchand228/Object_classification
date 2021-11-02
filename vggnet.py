import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from glob import glob
#import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def vgg16():
    IMAGE_SIZE = [224, 224]
    #Give dataset path
    train_path = 'train'
    test_path = 'test'
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False
    # useful for getting number of classes
    folders = glob('train/*')
    print(len(folders))
    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    # Data Augmentation
    train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    # Data Augmentation
    test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    # Make sure you provide the same target size as initialied for the image size
    train_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



#lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

#num_epochs = 1000
#num_batch_size = 32

    checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                               verbose=1, save_best_only=True)

    callbacks = [checkpoint, lr_reducer]

    start = datetime.now()

    model.fit_generator(
  train_set,
  validation_data=test_set,
  epochs=30,
  steps_per_epoch=10,
  validation_steps=32,
    callbacks=callbacks ,
    verbose=1)
    model.save("trained_vgg_flowers.h5")
    duration = datetime.now() - start 
    print("Training completed in time: ", duration)                                                                                                   

if __name__ =="__main__":
    vgg16()