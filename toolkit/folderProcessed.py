import os


def fp(target_path):
    # list of directories
    
    #labels=[]
    #images=numpy.zeros((thr_counter.loadChars74k(path,num_classes,num_samples),50,50,3))
    
    
    #Get directories in given folder
    dirlist = [ item for item in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, item)) ]
    #dirlist = dirlist[0:num_classes]
    #dirlist = dirlist[0:10]
    print(dirlist)
    
    
    dir_count=len(dirlist)
    
    for i in range(dir_count):        
        padded_name= format(str(i+1), "0>3s")
        #os.mkdir(target_path+os.sep+"Sample"+padded_name)
        os.rename(target_path+os.sep+dirlist[i],target_path+os.sep+"Sample"+padded_name)
        
        
        
    """
    os.rename(target_path+os.sep+"Sample"+padded_name,target_dir+os.sep+getFlename(TRAIN_DIR, file, counter))
    
    # for each subfolder, open all files, append to list of images x and set path as label in y
    start_ix=0
    for subfolder in dirlist:   
        imagePaths = glob.glob(path + '\\' + subfolder +'\\*.png')
        imagePaths = imagePaths[0:num_samples]
        print(subfolder)
        im_array = numpy.array( [numpy.array(Image.open(imagePath).convert('RGB'), 'f') for imagePath in imagePaths] )
        
        images[start_ix:start_ix+len(im_array)] = im_array
        start_ix += len(im_array)
        for imagePath in imagePaths:
            labels.append(int(subfolder[6:])-1)
    print(len(labels))
    return images, to_categorical(num_classes,labels)
    """
if __name__ == "__main__":
    
    fp("C:\\MLDatabases\\kannadaHandWrittenImages\\resized\\train")