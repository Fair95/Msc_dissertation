from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from constants import seed,batch_size,height_shift_range,width_shift_range,vertical_flip,rotation_range,zoom_range,horizontal_flip
def data_generator(X,y):
    data_gen_args = dict(height_shift_range=height_shift_range,
                         width_shift_range=width_shift_range,
                         rotation_range=rotation_range,
                         zoom_range=zoom_range,
                         vertical_flip=vertical_flip,
                         horizontal_flip=horizontal_flip)
        
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
                         
    image_generator = image_datagen.flow(X,seed=seed,batch_size=batch_size)
    mask_generator = mask_datagen.flow(y,seed=seed,batch_size=batch_size)
                         
    train_generator = zip(image_generator, mask_generator)
                         
    return train_generator

if __name__ == '__main__':
    data_generator(X,y)

