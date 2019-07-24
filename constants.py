seed = 1
inner = False
# Data paths
train_raw_path = 'Train_Raw'
train_mask_path = 'Training'
valid_raw_path = 'Valid_Raw'
valid_mask_path = 'Testing'

# Image size
image_rows = 512
image_cols = 512

img_rows = 512
img_cols = 512

threshold = 0.5

# Train parameters
lr = 1e-5
batch_size = 20
apply_augmentation = False
transfer_learning = False

# Uncertainty parameters
sample_steps = 50

# Data augmentation parameters
height_shift_range = 0.2
width_shift_range = 0.2
rotation_range=30
zoom_range=0.2
horizontal_flip=True
vertical_flip=True

