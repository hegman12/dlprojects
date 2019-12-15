import numpy as np
import tensorflow as tf
import os,ntpath
import scipy.misc as misc
from math import ceil
import json
import gen_graph,Model_Interpretor
import matplotlib.pyplot as plt

def calculate_shapes(filters,strides,padding):
    
    if padding=="SAME":
        l1_out_height = ceil(50. / float(strides[0][1]))
        l1_out_width  = ceil(50. / float(strides[0][2]))
        
        l2_out_height = ceil(l1_out_height / float(strides[1][1]))
        l2_out_width  = ceil(l1_out_width / float(strides[1][2]))

        l3_out_height = ceil(l2_out_height / float(strides[2][1]))
        l3_out_width  = ceil(l2_out_width / float(strides[2][2]))

        l4_out_height = ceil(l3_out_height / float(strides[3][1]))
        l4_out_width  = ceil(l3_out_width / float(strides[3][2]))    
        
        
    else:
        l1_out_height = ceil(float(50. - filters[0][0] + 1) / float(strides[0][1]))
        l1_out_width  = ceil(float(50. - filters[0][1] + 1) / float(strides[0][2]))

        l2_out_height = ceil(float(l1_out_height - filters[1][0] + 1) / float(strides[1][1]))
        l2_out_width  = ceil(float(l1_out_width - filters[1][1] + 1) / float(strides[1][2]))
        
        l3_out_height = ceil(float(l2_out_height - filters[2][0] + 1) / float(strides[2][1]))
        l3_out_width  = ceil(float(l2_out_width - filters[2][1] + 1) / float(strides[2][2]))

        l4_out_height = ceil(float(l3_out_height - filters[3][0] + 1) / float(strides[3][1]))
        l4_out_width  = ceil(float(l3_out_width - filters[3][1] + 1) / float(strides[3][2]))
    
    return (l1_out_height,l1_out_width,l2_out_height,l2_out_width,l3_out_height,l3_out_width,l4_out_height,l4_out_width)


def inference(model_file=None,img_file=None,lr=0.001,dropout_rate=1.0):
    #np.set_printoptions(threshold=np.nan)
    true_label=1
    model_file="C:\\MLDatabases\\kannadaHandWrittenImages\\tf_10E_400S_4L_SoftCrossEnt_normalisedV1_10C.model_config"
    img_file="C:\\MLDatabases\\kannadaHandWrittenImages\\test\\Sample001\\Sample0014_img001-019.png"
    with open(model_file,"r") as f:
        params=json.load(f)
    
    sh=calculate_shapes(params["filters"],params["strides"],params["padding"])
    print(sh)
    for s in sh:
        assert(s>0),"These filters/strides doesnt work with convolution layers"
    
    
    #learning_rate=tf.placeholder("float32", (), name="learning_rate")
    #drop_rate=tf.placeholder("float32", (), name="drop_rate")
    model_file=params["model_file"]
    tf.reset_default_graph()
    soft_max,new_saver,x=gen_graph.build_graph(params["strides"],params["filters"],params["padding"],sh,params["fcl"],params["dropout_rate"],params["num_classes"],model_file)

    #print("logits",logits)
    
    #x_data=misc.imread(img_file)
    x_data=Model_Interpretor.get_image(img_file)
    size=len(x_data)
    #print(x_data[0,0:10,0:10,:][x_data[0,0:10,0:10,:]==0.])

    #x_data=x_data[np.newaxis,:,:,:]
    #if model_file is None:
        
    print("image file read")
    probs=[]
    
    #for i in range(size):    
    
        #new_saver = tf.train.import_meta_graph(model_file + '.meta' , clear_devices=True)     
    sess=tf.Session()
    new_saver.restore(sess, model_file)
    for i in range(size):
        s_max=sess.run(soft_max,feed_dict={x:x_data[i][np.newaxis,:,:,:]})
        probs.append(s_max[0][true_label-1])
            #tf.reset_default_graph()
    sess.close()  
    idx_to_char={0:"A",1:"B",2:"C",3:"D",4:"E"}
    #print(probs)
    probs=np.array(probs)
    it=np.argmin(probs)
    plt.imshow(x_data[it])
    plt.show()    
    print("Iteration",it)
    plt.plot(probs)
    plt.show()

    #result_label.config(text="The selected image is: "+idx_to_char[np.argmax(s_max)] +"     ["+str(s_max.ravel())+"]" )
    
    
if "__main__"==__name__:
    inference()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    