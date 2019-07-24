from __future__ import print_function

from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from loss import *

from keras import regularizers
from constants import img_rows, img_cols, lr


#Override Dropout. Make it able at test time.
#def call(self, inputs, training=None):
#    if 0. < self.rate < 1.:
#        noise_shape = self._get_noise_shape(inputs)
#        def dropped_inputs():
#            return K.dropout(inputs, self.rate, noise_shape,
#                             seed=self.seed)
#        if (training):
#            return K.in_train_phase(dropped_inputs, inputs, training=training)
#        else:
#            return K.in_test_phase(dropped_inputs, inputs, training=None)
#    return inputs
#
#Dropout.call = call

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[3] - refer.get_shape()[3]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)
            
    return (ch1, ch2), (cw1, cw2)

def get_unet(dropout):
    inputs = Input((1,img_rows, img_cols))
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU(alpha = 0.1)(conv1)
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = LeakyReLU(alpha = 0.1)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU(alpha = 0.1)(conv2)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = LeakyReLU(alpha = 0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU(alpha = 0.1)(conv3)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = LeakyReLU(alpha = 0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU(alpha = 0.1)(conv4)
    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = LeakyReLU(alpha = 0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU(alpha = 0.1)(conv5)
    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU(alpha = 0.1)(conv5)
    
    if dropout:
        conv5 = Dropout(0.5)(conv5,training=True)

    up6 = UpSampling2D(size = (2,2))(conv5)
    ch, cw = get_crop_shape(conv4, up6)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    merge6 = concatenate([up6,crop_conv4], axis = 1)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU(alpha = 0.1)(conv6)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha = 0.1)(conv6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    ch, cw = get_crop_shape(conv3, up7)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    merge7 = concatenate([up7,crop_conv3], axis = 1)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU(alpha = 0.1)(conv7)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha = 0.1)(conv7)

    up8 = UpSampling2D(size = (2,2))(conv7)
    ch, cw = get_crop_shape(conv2, up8)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    merge8 = concatenate([up8,crop_conv2], axis = 1)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU(alpha = 0.1)(conv8)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha = 0.1)(conv8)

    up9 = UpSampling2D(size = (2,2))(conv8)
    ch, cw = get_crop_shape(conv1, up9)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    merge9 = concatenate([up9,crop_conv1], axis = 1)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU(alpha = 0.1)(conv9)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha = 0.1)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=weighted_bce_dice_loss, metrics=['binary_crossentropy',dice_coef])
    return model

def get_unet_shallow(dropout):
    l2_reg=1e-4
    inputs = Input((1,img_rows, img_cols))
    conv1 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv1)
    conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv2)
    conv2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv3)
    conv3 = Dropout(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv4)
    conv4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv5)
    
    if dropout:
        conv5 = Dropout(0.5)(conv5,training=True)
    else:
        conv5 = Dropout(0.1)(conv5)
    
    up6 = UpSampling2D(size = (2,2))(conv5)
    ch, cw = get_crop_shape(conv4, up6)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    merge6 = concatenate([up6,crop_conv4], axis = 1)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(merge6)
    conv6 = Dropout(0.1)(conv6)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv6)
    conv6 = Dropout(0.1)(conv6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    ch, cw = get_crop_shape(conv3, up7)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    merge7 = concatenate([up7,crop_conv3], axis = 1)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(merge7)
    conv7 = Dropout(0.1)(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv7)
    conv7 = Dropout(0.1)(conv7)

    up8 = UpSampling2D(size = (2,2))(conv7)
    ch, cw = get_crop_shape(conv2, up8)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    merge8 = concatenate([up8,crop_conv2], axis = 1)
    conv8 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(merge8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv8)
    conv8 = Dropout(0.1)(conv8)

    up9 = UpSampling2D(size = (2,2))(conv8)
    ch, cw = get_crop_shape(conv1, up9)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    merge9 = concatenate([up9,crop_conv1], axis = 1)
    conv9 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(merge9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer = regularizers.l2(l2_reg))(conv9)
    conv9 = Dropout(0.1)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=weighted_dice_loss, metrics=['binary_crossentropy',dice_coef])
    return model

def get_unet_simple():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up6 = UpSampling2D(size = (2,2))(conv5)
    merge6 = concatenate([up6,conv4], axis = 1)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = UpSampling2D(size = (2,2))(conv6)
    merge7 = concatenate([up7,conv3], axis = 1)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = UpSampling2D(size = (2,2))(conv7)
    merge8 = concatenate([up8,conv2], axis = 1)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = UpSampling2D(size = (2,2))(conv8)
    merge9 = concatenate([up9,conv1], axis = 1)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=weighted_dice_loss, metrics=['binary_crossentropy',dice_coef])
    return model


def get_unet_inner(dropout):
    inputs = Input((1,img_rows, img_cols))
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU(alpha = 0.1)(conv1)
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = LeakyReLU(alpha = 0.1)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU(alpha = 0.1)(conv2)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = LeakyReLU(alpha = 0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU(alpha = 0.1)(conv3)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = LeakyReLU(alpha = 0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU(alpha = 0.1)(conv4)
    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = LeakyReLU(alpha = 0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU(alpha = 0.1)(conv5)
    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU(alpha = 0.1)(conv5)
    
    if dropout:
        conv5 = Dropout(0.5)(conv5,training=True)
        
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    poll5_flat = Flatten()(pool5)
    dense1 = Dense(1024, activation='relu')(poll5_flat)
    dense2 = Dense(512, activation='relu')(dense1)
    dense3 = Dense(1, activation='sigmoid', name='has_mask_output')(dense1)
    
    up6 = UpSampling2D(size = (2,2))(conv5)
    ch, cw = get_crop_shape(conv4, up6)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    merge6 = concatenate([up6,crop_conv4], axis = 1)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU(alpha = 0.1)(conv6)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha = 0.1)(conv6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    ch, cw = get_crop_shape(conv3, up7)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    merge7 = concatenate([up7,crop_conv3], axis = 1)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU(alpha = 0.1)(conv7)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha = 0.1)(conv7)

    up8 = UpSampling2D(size = (2,2))(conv7)
    ch, cw = get_crop_shape(conv2, up8)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    merge8 = concatenate([up8,crop_conv2], axis = 1)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU(alpha = 0.1)(conv8)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha = 0.1)(conv8)

    up9 = UpSampling2D(size = (2,2))(conv8)
    ch, cw = get_crop_shape(conv1, up9)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    merge9 = concatenate([up9,crop_conv1], axis = 1)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU(alpha = 0.1)(conv9)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha = 0.1)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    conv10_flat = Flatten()(conv10)
    dense3_repeat = Flatten()(RepeatVector(img_rows * img_cols)(dense3))
    merge_output = Multiply()([conv10_flat, dense3_repeat])
    out = Reshape((1, img_rows, img_cols), name='main_output')(merge_output)

    model = Model(input=inputs, output=[out, dense3])
    model.compile(optimizer=Adam(lr=1e-5),loss={'main_output': weighted_bce_dice_loss, 'has_mask_output': 'binary_crossentropy'},metrics={'main_output': dice_coef, 'has_mask_output': 'accuracy'})
    return model
