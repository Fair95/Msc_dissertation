from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras
import random
import os
import unet
from keras import backend as K
from data_new import load_train_data
from constants import *
from loss import *
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

smooth = 1.

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0
    
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1
        
        self.valid.append(logs.get('val_dice_coef'))
        
        with open("runs/history.txt" , "a") as fout:
            fout.write("train\t%d\t%.4f\n" % (self.lastiter, self.losses[-1]))
            fout.write("valid\t%d\t%.4f\n" % (self.lastiter, self.valid[-1]))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1
        with open("runs/history.txt", "a") as fout:
            fout.write("train\t%d\t%.4f\n" % (self.lastiter, self.losses[-1]))



def data_generator(X,y):
    data_gen_args = dict(rotation_range=rotation_range,
                         zoom_range=zoom_range,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow(X,seed=seed,batch_size=50)
    mask_generator = mask_datagen.flow(y,seed=seed,batch_size=50)
    
    train_generator = zip(image_generator, mask_generator)

    return train_generator


def train(epochs):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet.get_unet(dropout=False)
    if transfer_learning:
        model.load_weights("transfer_learning/new.hdf5", by_name=True)
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    X,y = load_train_data()
    print(X.shape)
    imgs_train,imgs_valid,imgs_mask_train,imgs_mask_valid = train_test_split(X,y,test_size=0.2,random_state = 1)
    
    print('-'*30)
    print('Evaluating transfer learning.')
    print('-'*30)
    
    valloss = model.evaluate(x = imgs_valid, y = imgs_mask_valid, batch_size=10, verbose=0)
    with open("runs/history.txt" , "a") as fout:
        fout.write("valid\t%d\t%.4f\n" % (0, valloss[1]))
    filepath="runs/weights_%d_%d.hdf5" % (img_rows,img_cols)
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor="val_dice_coef", mode="max")
    
    history = LossHistory()
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    if apply_augmentation:
        model.fit_generator(data_generator(imgs_train,imgs_mask_train), validation_data = (imgs_valid, imgs_mask_valid),
                  steps_per_epoch= 100, epochs=epochs, verbose=1, callbacks=[history, checkpoint])
    else:
        model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, validation_data=(imgs_valid,imgs_mask_valid),
                  epochs=epochs, verbose=1, callbacks=[history, checkpoint])
    return model




if __name__ == '__main__':
    train(20)
