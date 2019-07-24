from __future__ import print_function

import os
import numpy as np

import cv2
from sklearn.model_selection import train_test_split

from constants import *

def label_by_threshold(threshold,input_seg):
    output_seg = input_seg >= threshold
    output_seg = output_seg.astype('float32')
    return output_seg

def preprocessor(input_img):
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
    return output_img

def create_train_data():
    print('-'*30)
    print('Reading training images...')
    print('-'*30)
    images = sorted(os.listdir(train_raw_path))
    masks = sorted(os.listdir(train_mask_path))
    total = len(images)
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask1 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask2 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    i = 0
    for image_name in images:
        if 'image' in image_name:
            ori = cv2.imread(os.path.join(train_raw_path, image_name), cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(os.path.join(train_mask_path, image_name.split('.')[0]+'-srf-grader1.png'), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(os.path.join(train_mask_path, image_name.split('.')[0]+'-srf-grader2.png'), cv2.IMREAD_GRAYSCALE)
            if ori.shape[1]<512:
                ori = np.concatenate((ori, cv2.flip( ori, 1 )), axis=1)
                mask1 = np.concatenate((mask1, cv2.flip( mask1, 1 )), axis=1)
                mask2 = np.concatenate((mask2, cv2.flip( mask2, 1 )), axis=1)
            ym = np.argmax(np.sum(ori,axis=1))
            xm = int(ori.shape[1]/2)
            
            y0 = int(ym - img_rows / 2)
            y1 = int(ym + img_rows / 2)
            x0 = int(xm - img_cols / 2)
            x1 = int(xm + img_cols / 2)
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if y1 > ori.shape[0]:
                y0 -= y1-ori.shape[0]
                y0 = max(0,y0)
                y1 = ori.shape[0]
            if x0 < 0:
                x0 = 0
            if x1 > ori.shape[1]:
                x1 = ori.shape[1]
        
            img = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask1 = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask2 = np.zeros((image_rows, image_cols), dtype="uint8")
            
            img[0:y1-y0, 0:x1-x0] = ori[y0:y1,x0:x1]
            img_mask1[0:y1-y0, 0:x1-x0] = mask1[y0:y1,x0:x1]
            img_mask2[0:y1-y0, 0:x1-x0] = mask2[y0:y1,x0:x1]
            
            img = np.array([img])
            imgs[i] = img
            img_mask1 = np.array([img_mask1])
            imgs_mask1[i] = img_mask1
            img_mask2 = np.array([img_mask2])
            imgs_mask2[i] = img_mask2
            i += 1
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
    print('Done: {0}/{1} images'.format(total, total))

    print('Creating final mask...')
    imgs_mask = cv2.add(0.5*imgs_mask1,0.5*imgs_mask2)
    #imgs_mask = label_by_threshold(threshold,imgs_mask)
    print('Done')
    print('Saving to npy files...')
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Done')

#def create_train_data():
#    print('-'*30)
#    print('Reading training raw images...')
#    print('-'*30)
#    images = sorted(os.listdir(train_raw_path))
#    masks = sorted(os.listdir(train_mask_path))
#    total = len(images)
#    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask1 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask2 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    i = 0
#    for image_name in images:
#        if 'image' in image_name:
#            img = cv2.imread(os.path.join(train_raw_path, image_name), cv2.IMREAD_GRAYSCALE)
#            img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_LINEAR)
#            img = np.array([img])
#            imgs[i] = img
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('-'*50)
#    print('Reading training mask images from grader1...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader1' in image_mask_name:
#            img_mask1 = cv2.imread(os.path.join(train_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask1 = cv2.resize(img_mask1, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask1 = np.array([img_mask1])
#            imgs_mask1[i] = img_mask1
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#    print('-'*50)
#    print('Reading training mask images from grader2...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader2' in image_mask_name:
#            img_mask2 = cv2.imread(os.path.join(train_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask2 = cv2.resize(img_mask2, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask2 = np.array([img_mask2])
#            imgs_mask2[i] = img_mask2
#
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#
#
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('Creating final mask...')
#    imgs_mask = cv2.add(0.5*imgs_mask1,0.5*imgs_mask2)
#   # imgs_mask = label_by_threshold(threshold,imgs_mask)
#    print('Done')
#    print('Saving to npy files...')
#    np.save('imgs_train.npy', imgs)
#    np.save('imgs_mask_train.npy', imgs_mask)
#    print('Done')
#
def load_train_data():
    print('-'*30)
    print('Loading training raw images...')
    print('-'*30)
    X_train = np.load('imgs_train.npy')
    y_train = np.load('imgs_mask_train.npy')
   # X_train = preprocessor(X_train)
   # y_train = preprocessor(y_train)
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

def create_valid_data():
    print('-'*30)
    print('Reading valid images...')
    print('-'*30)
    images = sorted(os.listdir(valid_raw_path))
    masks = sorted(os.listdir(valid_mask_path))
    total = len(images)
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask1 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask2 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask3 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask4 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    weight = 1/4
    i = 0
    for image_name in images:
        if 'image' in image_name:
            ori = cv2.imread(os.path.join(valid_raw_path, image_name), cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(os.path.join(valid_mask_path, image_name.split('.')[0]+'-srf-grader1-1.png'), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(os.path.join(valid_mask_path, image_name.split('.')[0]+'-srf-grader2-1.png'), cv2.IMREAD_GRAYSCALE)
            mask3 = cv2.imread(os.path.join(valid_mask_path, image_name.split('.')[0]+'-srf-grader3-1.png'), cv2.IMREAD_GRAYSCALE)
            mask4 = cv2.imread(os.path.join(valid_mask_path, image_name.split('.')[0]+'-srf-grader4-1.png'), cv2.IMREAD_GRAYSCALE)
            if ori.shape[1]<512:
                ori = np.concatenate((ori, cv2.flip( ori, 1 )), axis=1)
                mask1 = np.concatenate((mask1, cv2.flip( mask1, 1 )), axis=1)
                mask2 = np.concatenate((mask2, cv2.flip( mask2, 1 )), axis=1)
                mask3 = np.concatenate((mask3, cv2.flip( mask3, 1 )), axis=1)
                mask4 = np.concatenate((mask4, cv2.flip( mask4, 1 )), axis=1)
            
            ym = np.argmax(np.sum(ori,axis=1))
            xm = int(ori.shape[1]/2)
            y0 = int(ym - img_rows / 2)
            y1 = int(ym + img_rows / 2)
            x0 = int(xm - img_cols / 2)
            x1 = int(xm + img_cols / 2)
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if y1 > ori.shape[0]:
                y0 -= y1-ori.shape[0]
                y0 = max(0,y0)
                y1 = ori.shape[0]
            if x0 < 0:
                x0 = 0
            if x1 > ori.shape[1]:
                x1 = ori.shape[1]
            
            img = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask1 = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask2 = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask3 = np.zeros((image_rows, image_cols), dtype="uint8")
            img_mask4 = np.zeros((image_rows, image_cols), dtype="uint8")
            
            img[0:y1-y0, 0:x1-x0] = ori[y0:y1,x0:x1]
            img_mask1[0:y1-y0, 0:x1-x0] = mask1[y0:y1,x0:x1]
            img_mask2[0:y1-y0, 0:x1-x0] = mask2[y0:y1,x0:x1]
            img_mask3[0:y1-y0, 0:x1-x0] = mask3[y0:y1,x0:x1]
            img_mask4[0:y1-y0, 0:x1-x0] = mask4[y0:y1,x0:x1]
            
            img = np.array([img])
            imgs[i] = img
            img_mask1 = np.array([img_mask1])
            imgs_mask1[i] = img_mask1
            img_mask2 = np.array([img_mask2])
            imgs_mask2[i] = img_mask2
            img_mask3 = np.array([img_mask3])
            imgs_mask3[i] = img_mask3
            img_mask4 = np.array([img_mask4])
            imgs_mask4[i] = img_mask4
            i += 1
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
    print('Done: {0}/{1} images'.format(total, total))



    print('Creating final mask...')
    imgs_mask = cv2.add(cv2.add(weight*imgs_mask1,weight*imgs_mask2),cv2.add(weight*imgs_mask3,weight*imgs_mask4))
   #imgs_mask = label_by_threshold(threshold,imgs_mask)
    print('Done')
    
    print('Saving to npy files...')
    np.save('imgs_valid.npy', imgs)
    np.save('imgs_mask_valid.npy', imgs_mask)
    print('Done')

#def create_valid_data():
#    print('-'*30)
#    print('Reading valid raw images...')
#    print('-'*30)
#    images = sorted(os.listdir(valid_raw_path))
#    masks = sorted(os.listdir(valid_mask_path))
#    total = len(images)
#    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask1 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask2 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask3 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    imgs_mask4 = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
#    weight = 1/4
#    i = 0
#    for image_name in images:
#        if 'image' in image_name:
#            img = cv2.imread(os.path.join(valid_raw_path, image_name), cv2.IMREAD_GRAYSCALE)
#            img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_LINEAR)
#            img = np.array([img])
#            imgs[i] = img
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('-'*50)
#    print('Reading validate mask images from grader1...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader1-1' in image_mask_name:
#            img_mask1 = cv2.imread(os.path.join(valid_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask1 = cv2.resize(img_mask1, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask1 = np.array([img_mask1])
#            imgs_mask1[i] = img_mask1
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('-'*50)
#    print('Reading validate mask images from grader2...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader2-1' in image_mask_name:
#            img_mask2 = cv2.imread(os.path.join(valid_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask2 = cv2.resize(img_mask2, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask2 = np.array([img_mask2])
#            imgs_mask2[i] = img_mask2
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('-'*50)
#    print('Reading validate mask images from grader3...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader3-1' in image_mask_name:
#            img_mask3 = cv2.imread(os.path.join(valid_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask3 = cv2.resize(img_mask3, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask3 = np.array([img_mask3])
#            imgs_mask3[i] = img_mask3
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#    print('-'*50)
#    print('Reading validate mask images from grader4...')
#    print('-'*50)
#    i = 0
#    for image_mask_name in masks:
#        if 'grader4-1' in image_mask_name:
#            img_mask4 = cv2.imread(os.path.join(valid_mask_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
#            img_mask4 = cv2.resize(img_mask4, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
#            img_mask4 = np.array([img_mask4])
#            imgs_mask4[i] = img_mask4
#            i += 1
#            if i % 100 == 0:
#                print('Done: {0}/{1} images'.format(i, total))
#    print('Done: {0}/{1} images'.format(total, total))
#
#    print('Creating final mask...')
#    imgs_mask = cv2.add(cv2.add(weight*imgs_mask1,weight*imgs_mask2),cv2.add(weight*imgs_mask3,weight*imgs_mask4))
#imgs_mask = label_by_threshold(threshold,imgs_mask)
#    print('Done')
#
#    print('Saving to npy files...')
#    np.save('imgs_valid.npy', imgs)
#    np.save('imgs_mask_valid.npy', imgs_mask)
#    print('Done')
#
def load_valid_data():
    print('-'*30)
    print('Loading valid raw images...')
    print('-'*30)
    X_valid = np.load('imgs_valid.npy')
    y_valid = np.load('imgs_mask_valid.npy')
   # X_valid = preprocessor(X_valid)
   # y_valid = preprocessor(y_valid)
    print('Loading done')
    
    print('-'*30)
    print('preprocessing...')
    print('-'*30)
    X_valid = X_valid.astype('float32')
    mean = np.mean(X_valid)  # mean for data centering
    std = np.std(X_valid)  # std for data normalization
    
    X_valid -= mean
    X_valid /= std
    
    y_valid = y_valid.astype('float32')
    y_valid /= 255.  # scale masks to [0, 1]
    print('Loading done')
    
    return X_valid, y_valid

def load_data(downsample = True,use_neg=True):
    ''' Rebalace the data'''
    np.random.seed(seed = seed)
    imgs_train,imgs_mask_train = load_train_data()
    imgs_valid,imgs_mask_valid = load_valid_data()
    
    ind = np.where(np.sum(imgs_mask_train,axis=(1,2,3)))
    X = imgs_train[ind]
    y = imgs_mask_train[ind]

    train_ind = np.random.choice(ind[0],int(np.floor(len(ind[0])*0.8)),replace=False)
    valid_ind = np.setdiff1d(ind[0],train_ind)
    
    num_train = len(train_ind)
    num_valid = len(valid_ind)
    
    neg = np.setdiff1d(np.array(range(0,574)),ind[0])
    neg_train_ind = np.random.choice(neg,num_train,replace=False)
    neg_valid_ind = np.random.choice(np.setdiff1d(neg,neg_train_ind),num_valid,replace=False)
    
    # include images with no SRF segmentation
    if use_neg:
        final_train_ind = np.array(np.concatenate((train_ind,neg_train_ind)))
        final_valid_ind = np.array(np.concatenate((valid_ind,neg_valid_ind)))
    else:
        final_train_ind = train_ind
        final_valid_ind = valid_ind

    X_train = imgs_train[final_train_ind]
    X_valid = imgs_train[final_valid_ind]
    y_train = imgs_mask_train[final_train_ind]
    y_valid = imgs_mask_train[final_valid_ind]

    X_test = imgs_valid
    y_test = imgs_mask_valid
    
    # train/valid 8:2 + test (neg sample to balance)
    # else train/valid only with no negtive samples (98:15)
    if downsample:
        return X_train,y_train,X_valid,y_valid,X_test,y_test
    else:
        all_ind = np.array(range(0,imgs_train.shape[0]))
        train_ind = np.random.choice(all_ind,int(np.floor(len(all_ind))*0.8),replace=False)
        valid_ind = np.setdiff1d(all_ind,train_ind)
        X_train = imgs_train[train_ind]
        X_valid = imgs_train[valid_ind]
        y_train = imgs_mask_train[train_ind]
        y_valid = imgs_mask_train[valid_ind]        
        return X_train,y_train,X_valid,y_valid,X_test,y_test

if __name__ == '__main__':
    create_train_data()
    create_valid_data()
