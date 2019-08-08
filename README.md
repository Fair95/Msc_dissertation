# MSc Disseartation

## This is the repository for my MSc dissertation

### Setup
Please put the images in the corresponding folders.  
- Testing: Mask images for testing. The name format should look like "image-0-srf-grader1-1.png"
- Valid_Raw: Raw images for testing. The name format should look like "image-0.png"
- Training: Mask images for training. The name format should look like "image-0-srf-grader1-1.png"
- Train_Raw: Raw images for training. The name format should look like "image-0.png"

### Runing
Use constants.py to change different setting such as enabling/disabling augmentation.  The augmentation is turned off in default. Please turn it on to obtain the best result (but this would increase the training time significantly).  
Use unet.py to test different models.  The basic model corresponds to get_unet_shallow() in our study. The nested model corresponds to get_unet_inner(). The original U-net model corresponds to get_unet(). 
Use loss.py to test different loss function. The best loss function is weighted_bce_dice_loss().  
To run the model, use python3 main.py.   

### Testing the model using public available dataset.
We have tested our model configuration using public available dataset.  
Please download it following [this](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) and put the data into **ultrasound-nerve-segmentation** folder.  Note that since the masks for testing images are not provided by the data provider. Train/valid/test split is done using training images only.  
Modify constants.py file to change the image_rows and image_cols to 420, 580 respectively. And change img_row, img_cols to 96, 128 respectively.  
Run data_new.py to read the image data into npy file.  
Then run train_new.py to train using ultrasound nerve segmentation data.  
