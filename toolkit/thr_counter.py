import glob
import os

def loadChars74k(path,num_classes,num_samples):
    # list of directories
    
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]
    dirlist = dirlist[0:num_classes]
    #dirlist = dirlist[0:10]
    counter=0
    # for each subfolder, open all files, append to list of images x and set path as label in y
    for subfolder in dirlist:   
        imagePaths = glob.glob(path + '\\' + subfolder +'\\*.png')
        imagePaths = imagePaths[0:num_samples]
        counter += len(imagePaths)
    print(counter)
    return counter

if __name__ == "__main__":
    
    loadChars74k("C:\\MLDatabases\\kannadaRescalatedImages\\kannadahandWritten\\good")

