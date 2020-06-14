import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from image_patch_functions import *

def make_predictions(impaths, maskpaths, dirName, mode, cnn, patchsize=32, debug=False, minibatchsize=2000):
  """
  Produces the segmentation probability arrays and the corresponding
  PNG images. Note: predictions need to be thresholded before submitted

  mode: 'train', 'val' or 'test'
  cnn: the trained model
  """

  halfsize = int(patchsize/2)

  for j in range(len(impaths)):
      print(impaths[j])

      # Keep only green channel. Note that the scalling takes place in the paches
      image = np.array(PIL.Image.open(impaths[j]),dtype=np.int16)[:,:,1]
      mask = np.array(PIL.Image.open(maskpaths[j]),dtype=np.int16)

      image = np.pad(image,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
      mask = np.pad(mask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

      samples = np.nonzero(mask)
      probimage_thin = np.zeros(image.shape)
      probimage_thick = np.zeros(image.shape)
      probabilities_thin = np.empty((0,))
      probalilities_thick = np.empty((0,))
      for i in range(0,len(samples[0]),minibatchsize):
          print('{}/{} samples labelled'.format(i,len(samples[0])))

          if i+minibatchsize < len(samples[0]):
              batch = np.arange(i,i+minibatchsize)
          else:
              batch = np.arange(i,len(samples[0]))

          X = make2Dpatchestest(samples,batch,image,patchsize=patchsize)

          if debug:
              prob = np.random.rand(batch.shape[0], 2) # used for debugging
          else:
              prob = cnn.predict(X, batch_size=minibatchsize)

          probabilities_thin = np.concatenate((probabilities_thin,prob[:,1]))
          probalilities_thick = np.concatenate((probalilities_thick,prob[:,2]))

      for i in range(len(samples[0])):
          probimage_thin[samples[0][i],samples[1][i]] = probabilities_thin[i]
          probimage_thick[samples[0][i],samples[1][i]] = probalilities_thick[i]
      # Save the predictions
      if mode=="train":
          foldername = "//training_results//"
      elif mode=="val":
          foldername = "//validation_results//"
      elif mode=="test":
          foldername = "//test_results//"

      path_prob_thin = dirName + foldername + mode + "_probabilities_thin_{}".format(j+1)
      np.save(path_prob_thin, probimage_thin)
      
      path_prob_thick = dirName + foldername + mode + "_probabilities_thick_{}".format(j+1)
      np.save(path_prob_thick, probimage_thick)

      path_img_thin = dirName + foldername + 'thin' +mode + "{}.png".format(j+1)
      path_img_thick = dirName + foldername + 'thick' +mode + "{}.png".format(j+1)

      plt.figure()
      plt.imshow(probimage_thick,cmap='Greys_r')
      plt.axis('off')
      plt.savefig(path_img_thick)
      
      plt.figure()
      plt.imshow(probimage_thin,cmap='Greys_r')
      plt.axis('off')
      plt.savefig(path_img_thin)

  return
