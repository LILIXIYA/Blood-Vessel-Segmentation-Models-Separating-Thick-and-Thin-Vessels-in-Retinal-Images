# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:08:45 2019

@author: lxs
"""

'''
This py file do the transfering from Vgg16(trainied on imagenet). But please always restart python console before 
running this file, otherwise the name of each layer would change.
After transfering, the Unet cannot be fed for trainign immediatly.
The transfered parameters should first be freezed and other layers should be trained first.
Then transfered parameters could be freed and the whole network could be trained together.
Learning rate for 'freezed-training' should be 1e-5
'''
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.applications.vgg16 import VGG16

import numpy as np

def getUnet(pretrained_weights = None):
    input_size=(32, 32, 1)
    
    inputs = Input(input_size)
    #This part is the same with VGG16, so parameters could be transfered
    block1_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    block1_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2))(block1_conv2)
    
    block2_conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block1_pool)
    block2_conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block2_conv1)
    block2_pool = MaxPooling2D(pool_size=(2, 2))(block2_conv2)
    
    block3_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block2_pool)
    block3_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block3_conv1)
    block3_pool = MaxPooling2D(pool_size=(2, 2))(block3_conv2)
    
    block4_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block3_pool)
    block4_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(block4_conv1)
    
    drop4 = Dropout(0.5)(block4_conv2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #Below is the differnet part between U-Net and VGG16
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([block3_conv2,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([block2_conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([block1_conv2,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv11 = Flatten()(conv9)
    conv11 = Dense(120, activation='relu')(conv11)
    
    conv12 = Dense(84, activation='relu')(conv11)
    
    cov13 = Dense(3, activation='softmax')(conv12)
    
    cnn = Model(input = inputs, output = cov13)

    cnn.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return cnn

def getVGG():
    model = VGG16(weights='imagenet', include_top=False,input_shape=(64, 64, 3))
    return model

def mixVggUnet():
    #Load Unet and Vgg
    Unet = getUnet()
    Vgg = getVGG()
    #This layers share a same shape between Vgg and Unet
    fromVgg = dict()
    fromVgg['conv2d_1'] = 'block1_conv1'
    fromVgg['conv2d_2'] = 'block1_conv2'
    fromVgg['conv2d_3'] = 'block2_conv1'
    fromVgg['conv2d_4'] = 'block2_conv2'
    fromVgg['conv2d_5'] = 'block3_conv1'
    fromVgg['conv2d_6'] = 'block3_conv2'
    fromVgg['conv2d_7'] = 'block4_conv1'
    fromVgg['conv2d_8'] = 'block4_conv2'
    #transfer weights and bias.
    #For the input layer, only paraeters from green channals are transfered
    #Currently, we only use green channal for training, so the input layer 'Input'and first layer 'conv2d_1' is not trasfered
    #After transfering, the transfered layers should be freezend and training other layers for some period 
    #Then the whole network could be fed for training!
    #Learning rate is very tricky! Now it is setted to 1e-5.Maybe we can try some other values!
    for layersU in Unet.layers:
        print(layersU.name)
        if layersU.name in fromVgg:
            if layersU.name == 'conv2d_1':
                green_channal_weight = np.zeros_like(layersU.get_weights()[0])

                vgg_layer_name = fromVgg[layersU.name]
                full_channal = Vgg.get_layer(vgg_layer_name).get_weights()
            
                green_channal_weight[:,:,0,:] = full_channal[0][1]
                green_channal_bias = full_channal[1]
            
                layersU.set_weights([green_channal_weight,green_channal_bias])
            else:
                vgg_layer_name = fromVgg[layersU.name]
                weights = Vgg.get_layer(vgg_layer_name).get_weights()
                layersU.set_weights(weights)
                
            print('Load from Vgglayer'+vgg_layer_name)
            
    Unet.save(r'C:\Users\lxs\Desktop\Q3\medical imaging\project1\result\sample\Uet_transfer_notop.h5')
    
mixVggUnet()
    