import numpy
import glob
from PIL import Image
import os
import thr_counter
import numpy as np
from scipy import misc

def to_categorical(n_classes,y):
    
    return numpy.eye(n_classes)[y]

def loadChars74k(path,num_classes,num_samples):
    # list of directories
    
    labels=[]
    images=numpy.zeros((thr_counter.loadChars74k(path,num_classes,num_samples),50,50,3))
    
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]
    dirlist = dirlist[0:num_classes]
    #dirlist = dirlist[0:10]
    
    # for each subfolder, open all files, append to list of images x and set path as label in y
    start_ix=0
    for subfolder in dirlist:   
        imagePaths = glob.glob(path + '\\' + subfolder +'\\*.png')
        imagePaths = imagePaths[0:num_samples]
        print(subfolder)
        im_array = numpy.array( [misc.imresize(numpy.array(misc.imread(imagePath)),(50,50,3)) for imagePath in imagePaths] )
        
        images[start_ix:start_ix+len(im_array)] = im_array
        start_ix += len(im_array)
        for imagePath in imagePaths:
            labels.append(int(subfolder[6:])-1)
    print(len(labels))
    return images, to_categorical(num_classes,labels)

if __name__ == "__main__":
    
    loadChars74k("C:\\MLDatabases\\kannadaRescalatedImages\\kannadahandWritten\\good",None,None)

