from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import keras
import random
import os
import unet
from augmentation import data_generator
from keras import backend as K
from data import load_train_data,load_valid_data,load_data,label_by_threshold
from constants import *
import matplotlib.pyplot as plt
from uncertainty import *
from loss import *

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


def train(epochs):
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet.get_unet(dropout=True)
    if transfer_learning:
        model.load_weights("transfer_learning/new.hdf5", by_name=True)
    if inner:
        model = unet.get_unet_inner(dropout=True)
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

#    imgs_train,imgs_mask_train = load_train_data()
#
#    imgs_valid,imgs_mask_valid = load_valid_data()
#
#    ind = np.where(np.sum(imgs_mask_train,axis=(1,2,3)))
#
#    # Random sample negative samples
#    ran = np.setdiff1d(np.array(range(0,574)),ind[0])
#    choice = np.random.choice(ran,50)
#    final_ind = np.array(np.concatenate((ind[0], choice)))
#
#    imgs_mask_train=imgs_mask_train[final_ind]
#    imgs_train=imgs_train[final_ind]
#
#    ind = np.where(np.sum(imgs_mask_valid,axis=(1,2,3)))
#    # Random sample negative samples
#    ran = np.setdiff1d(np.array(range(0,60)),ind[0])
#    choice = np.random.choice(ran,10)
#    final_ind = np.array(np.concatenate((ind[0], choice)))
#
#    imgs_mask_valid = imgs_mask_valid[final_ind]
#    imgs_valid = imgs_valid[final_ind]
    X_train,y_train,X_valid,y_valid,X_test,y_test = load_data(downsample=False)
    y_train_has_mask = (y_train.reshape(y_train.shape[0], -1).max(axis=1) > 0) * 1
    y_valid_has_mask = (y_valid.reshape(y_valid.shape[0], -1).max(axis=1) > 0) * 1
    print('-'*30)
    print('Evaluating transfer learning.')
    print('-'*30)
    if not inner:
        valloss = model.evaluate(x = X_valid, y = y_valid, batch_size=10, verbose=0)
        with open("runs/history.txt" , "a") as fout:
            fout.write("valid\t%d\t%.4f\n" % (0, valloss[1]))
        filepath="runs/weights_%d_%d.hdf5" % (img_rows,img_cols)
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor="val_dice_coef", mode="max")
    else:
        filepath="runs/weights_%d_%d_inner.hdf5" %(img_rows,img_cols)
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor="val_main_output_dice_coef",mode="max")
    
    history = LossHistory()
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    if apply_augmentation:
        model.fit_generator(data_generator(X_train,y_train), validation_data = (X_valid, y_valid),
                  steps_per_epoch= 5*X_train.shape[0]//batch_size, shuffle = True, epochs=epochs, verbose=1, callbacks=[history, checkpoint])
    elif inner:
        model.fit(X_train, [y_train, y_train_has_mask], validation_data = (X_valid, [y_valid,y_valid_has_mask]), batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, callbacks=[checkpoint])
    else:
        model.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_valid, y_valid), shuffle = True,
                  epochs=epochs, verbose=1, callbacks=[history, checkpoint])
    
    return model, filepath


def plot_loss_history(model):
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    
    plt.savefig('history/loss.png')

def evaluate_model(model):
    X_train,y_train,X_valid,y_valid,X_test,y_test = load_data()
    y_test_has_mask = (y_test.reshape(y_test.shape[0], -1).max(axis=1) > 0) * 1
    if not inner:
        testloss = model.evaluate(x = X_test, y = y_test, verbose=0)
        fp=0
        fn=0
        threshold = 0.75
        y_pred = model.predict(X_test)
        np.save('pre.npy',y_pred)
        y_test_threshold = label_by_threshold(0.5,y_test)
        y_pred = label_by_threshold(threshold,y_pred)
        for i in range(0,y_test_has_mask.shape[0]):
            if (np.sum(y_test_has_mask[i])==0) & (np.sum(y_pred[i])!=0):
                fp+=1
            if (np.sum(y_test_has_mask[i])!=0) & (np.sum(y_pred[i])==0):
                fn+=1
        print('False positives:',fp)
        print('False negatives:',fn)
        print('dice coef at threshold %.2f :' % threshold,np_dice_coef(y_pred,y_test_threshold))
    print('-'*30)
    if inner:
        testloss = model.evaluate(x = X_test, y = [y_test,y_test_has_mask], verbose=0)
    for i in range(0,len(model.metrics_names)):
        print(model.metrics_names[i], ':', testloss[i])
    print('-'*30)
    print('Evaluating uncertainty...')
    print('-'*30)
    uncertainty = compute_uncertainty_map(model,X_test)
    print('Uncertainty:',np.mean(np.mean(np.sum(np.sum(uncertainty,axis=-1),axis=-1))))
    np.save('y_map.npy',uncertainty)
    print('Done')
    return testloss,uncertainty
          

if __name__ == '__main__':
    train(50)
