#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:40:25 2017

@author: johnnyhsieh
"""
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


height = 225
width = 225

base_image = 'content/brooklyn.jpg'
base_image = image.load_img(base_image, target_size=(height, width))
base_image = image.img_to_array(base_image).astype('float32')
base_image = np.expand_dims(base_image, axis=0)
#base_image = preprocess_input(base_image)
base_image[:, :, :, 0] -= 103.939
base_image[:, :, :, 1] -= 116.779
base_image[:, :, :, 2] -= 123.68
base_image = base_image[:, :, :, ::-1]



style_image = 'style/monet.jpg'
style_image = image.load_img(style_image, target_size=(height, width))
style_image = image.img_to_array(style_image).astype('float32')
style_image = np.expand_dims(style_image, axis=0)
#style_image = preprocess_input(style_image)
#infact this is the origianl process of preprocess_input of VGG166
style_image[:, :, :, 0] -= 103.939
style_image[:, :, :, 1] -= 116.779
style_image[:, :, :, 2] -= 123.68
style_image = style_image[:, :, :, ::-1]

import keras.backend as k
base_image = k.variable(base_image)
style_image = k.variable(style_image)
#placeholder will generate an tf.tensor channel 1, hight 300 wide 400 and RGB 
result_image = k.placeholder((1,height,width,3))

input_tensor = k.concatenate([base_image,style_image,result_image],axis = 0)

model = VGG16(input_tensor = input_tensor
                    ,weights = 'imagenet',include_top = False)
layers = dict([(layer.name, layer.output) for layer in model.layers])


content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1
loss = k.variable(0.)

def content_loss(content, combination):
    return k.sum(k.square(combination - content))

layer_features = layers['block3_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features,
                                      combination_features)
def gram_matrix(x):
    features = k.batch_flatten(k.permute_dimensions(x, (2, 0, 1)))
    gram = k.dot(features, k.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height*width
    return k.sum(k.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv1','block1_conv2','block2_conv1', 'block2_conv2','block3_conv1'
                  ,'block3_conv2','block3_conv3', 'block4_conv3',
                  'block5_conv3']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
    
    
def total_variation_loss(x):
    a = k.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = k.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return k.sum(k.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(result_image)

grads = k.gradients(loss, result_image)


outputs = [loss]
outputs += grads
f_outputs = k.function([result_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    fname = 'iteration_%d.jpg' % i 
    imsave(fname, x) 
    
    

    