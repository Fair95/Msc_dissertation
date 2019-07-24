from __future__ import print_function

from train import *

import unet

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

history, model_weights = train(50)

model = unet.get_unet(dropout=True)

if inner:
    model = unet.get_unet_inner(dropout=True)

model.load_weights(model_weights)

plot_loss_history(history)

test, uncertainty = evaluate_model(model)

