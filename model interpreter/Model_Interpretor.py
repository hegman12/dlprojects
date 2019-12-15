import numpy as np
from scipy import misc
import math
import matplotlib.pyplot as plt

def get_image(img_file):
    
    SKIP_LENGTH=10
    
    
    image=misc.imresize(misc.imread(img_file,mode="RGB"),(50,50,3))
        
    height,width,channel=image.shape
    x_count=math.ceil(height/SKIP_LENGTH)
    y_count=math.ceil(width/SKIP_LENGTH)
    
    size=x_count*y_count
    
    images=[]
    size_idx=0
    for i in range(0,height,SKIP_LENGTH):
        for j in range(0,width,SKIP_LENGTH):
            t=np.copy(image)
            t[i:i+SKIP_LENGTH,j:j+SKIP_LENGTH,0:3]=0.
            images.append(np.copy(t))
            size_idx += 1
    return np.array(images)

"""
import numpy as np
import matplotlib.pyplot as plt
import PIL
from cStringIO import StringIO


plt.imshow(np.random.random((20,20)))
buffer_ = StringIO()
plt.savefig(buffer_, format = "png")
buffer_.seek(0)
image = PIL.Image.open(buffer_)
ar = np.asarray(image)
buffer_.close()
"""