from __future__ import print_function

import os
import numpy as np

import cv2
from constants import *
data_path = 'ultrasound-nerve-segmentation'



def preprocessor(input_img):
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
    return output_img

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)
    
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])
        
        imgs[i] = img
        imgs_mask[i] = img_mask
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    print('-'*30)
    print('Loading training raw images...')
    print('-'*30)
    X_train = np.load('imgs_train.npy')
    y_train = np.load('imgs_mask_train.npy')
    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)
    print('Loading done')
    
    print('-'*30)
    print('preprocessing...')
    print('-'*30)
    X_train = X_train.astype('float32')
    
    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization
    
    X_train -= mean
    X_train /= std
    
    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    print('Done')
    
    return X_train, y_train



if __name__ == '__main__':
    create_train_data()
