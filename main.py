# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 09:30:15 2017

@author: pmoeskops
"""

'''
Adding UNet for this given template
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
 
import random
random.seed(0)
import glob
import PIL.Image
import copy

from image_patch_functions import *
from Unet_architecture import *
from segmentation_prediction import make_predictions
import Mutate_data

#inputs
impaths_all = glob.glob(r'.\training\images\*.tif')
trainingsetsize = 15
patchsize = 32
minibatchsize = 200
minibatches = 20000


#shuffle the images to take a random subset for training later
random.shuffle(impaths_all)

maskpaths_all = copy.deepcopy(impaths_all)
segpaths_all = copy.deepcopy(impaths_all)
segpaths_thin = copy.deepcopy(impaths_all)
segpaths_thick = copy.deepcopy(impaths_all)
#select the corresponding masks and segmentations
for i in range(len(impaths_all)):
    maskpaths_all[i] = impaths_all[i].replace('images','mask')
    maskpaths_all[i] = maskpaths_all[i].replace('.tif','_mask.gif')

    segpaths_thin[i] = impaths_all[i].replace('images','thin')
    segpaths_thin[i] = segpaths_thin[i].replace('training.tif','manual1.gif')
    
    segpaths_thick[i] = impaths_all[i].replace('images','thick')
    segpaths_thick[i] = segpaths_thick[i].replace('training.tif','manual1.gif')
    
    segpaths_all[i] = impaths_all[i].replace('images','1st_manual')
    segpaths_all[i] = segpaths_thick[i].replace('training.tif','manual1.gif')
    
print(impaths_all)
print(maskpaths_all)
print(segpaths_thin)
print(segpaths_thick)

#select the first 15 images as training set, the other 5 will be used for validation
impaths = impaths_all[:trainingsetsize]
maskpaths = maskpaths_all[:trainingsetsize]
segpaths_all = segpaths_all[:trainingsetsize]
segpaths_thin = segpaths_thin[:trainingsetsize]
segpaths_thick = segpaths_thick[:trainingsetsize]

#load the training images
images, masks, segmentations_thin, segmentations_thick, segmentations_all = loadImages(impaths,maskpaths,segpaths_thin, segpaths_thick, segpaths_all)
#images, masks, segmentations = Mutate_data.mutate_data(images, masks, segmentations)

print(images.shape)
print(masks.shape)
print(segmentations_thin.shape)
print(segmentations_thick.shape)
#pad the images with zeros to allow patch extraction at all locations
halfsize = int(patchsize/2)
images = np.pad(images,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
masks = np.pad(masks,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
segmentations_thin = np.pad(segmentations_thin,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
segmentations_thick = np.pad(segmentations_thick,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

#separately select the positive samples (vessel) and negative samples (background)
positivesamples_thin= np.nonzero(segmentations_thin)
positivesamples_thick = np.nonzero(segmentations_thick)
negativesamples = np.nonzero(masks-segmentations_thin)

print(len(positivesamples_thin[0]))
print(len(positivesamples_thick[0]))
print(len(negativesamples[0]))

trainnetwork = 0

#initialise the network
cnn = Unet(pretrained_weights = 1)
#and start training
if trainnetwork:
    losslist = []

    for i in range(minibatches):

        posbatch_thin = random.sample(list(range(len(positivesamples_thin[0]))),int(minibatchsize/4))
        posbatch_thick = random.sample(list(range(len(positivesamples_thick[0]))),int(minibatchsize/4))
        negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))

        Xpos_thin, Ypos_thin = make2Dpatches(positivesamples_thin,posbatch_thin,images,32,1)
        Xpos_thick, Ypos_thick = make2Dpatches(positivesamples_thick,posbatch_thick,images,32,2)
        Xneg, Yneg = make2Dpatches(negativesamples,negbatch,images,32,0)
        
        Xpos = np.vstack((Xpos_thin,Xpos_thick))
        Ypos = np.vstack((Ypos_thin,Ypos_thick))
        
        Xtrain = np.vstack((Xpos,Xneg))
        Ytrain = np.vstack((Ypos,Yneg))
     
        loss = cnn.train_on_batch(Xtrain,Ytrain)
        losslist.append(loss)
        print('Batch: {}'.format(i))
        print('Loss: {}'.format(loss))


    plt.close('all')
    plt.figure()
    plt.plot(losslist)

    cnn.save(r'.\3class.h5')

else:
    cnn = keras.models.load_model(r'.\3class.h5')
    # cnn = keras.models.load_model(r'.\experiments\Unet10000.h5') # for debugging


#### Use the trained network to predict ####
# Paths to images/masks
valimpaths = impaths_all[trainingsetsize:]
valmaskpaths = maskpaths_all[trainingsetsize:]

testimpaths = glob.glob(r'.\test\images\*.tif')
testmaskpaths = glob.glob(r'.\test\mask\*.gif')

trainimpaths = impaths_all[:trainingsetsize]
trainmaskpaths = maskpaths_all[:trainingsetsize]

# Create directory to store results
dirName = "3class"
try:
    # Create target directory
    os.mkdir(dirName)
    print("Directory ", dirName, " was created")
except FileExistsError:
    print("Directory", dirName, " already exists")
os.mkdir(dirName + "//training_results")
os.mkdir(dirName + "//validation_results")
os.mkdir(dirName + "//test_results")

debug = False # keep it False - othewise it does not use the trained network to predict
make_predictions(valimpaths, valmaskpaths, dirName, mode='val', cnn=cnn, patchsize=patchsize, debug=debug)
make_predictions(testimpaths, testmaskpaths, dirName, mode='test', cnn=cnn, patchsize=patchsize, debug=debug)
make_predictions(trainimpaths, trainmaskpaths, dirName, mode='train', cnn=cnn, patchsize=patchsize, debug=debug)
