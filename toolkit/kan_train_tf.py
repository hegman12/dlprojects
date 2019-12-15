import numpy as np
import tensorflow as tf
import math
import thr
import scipy.misc as misc
from PIL import ImageTk,Image
from math import ceil
import json


"""

This fuction calculates the matrix dimentions of intermediate convolution layer output.
This is required because from frontend user can pass any stride/filter values, we have to calculate the 
final convolutional layer shape based on the strides and filters.

"""

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


"""
This functiona creates the weights for network. We need to pass the matrix dimentions and it automatically creates the weights of requested dimentions.
Note that these weights are updated/learned during the training process.

This uses standard 0.03 standard deviation, we can change this according to our need.

"""

def create_weights(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0, stddev=0.03, dtype="float32", seed=10, name="conv1_w"),dtype="float32")


"""
This is the main functions where the training happens. This function is called from front end.
front end passes the UI elements which is used for UI update.

"""


def train(path,imglabels,weightLabels,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label,filters,strides,lr,dropout_rate,padding,fcl, num_classes, num_samples):
    
    """
    Resetting to graph for rerunning the module
    """
    
    tf.reset_default_graph()
    
    # Set seed for reproducibility
    tf.set_random_seed(100)
    train_accu,val_accu,test_accu="","",0
    
    try:
    
        """ set hyper parameters """
        BATCH_SIZE=50
        EPOCHS=10
        np.random.seed(10)
        
        num_classes = 657 if num_classes is None else num_classes
        num_samples = None if num_samples is None else num_samples
        
        sh=calculate_shapes(filters,strides,padding)
        print(sh)
        
        """ raise error if the calculated shares doesnt work """
        
        for s in sh:
            assert(s>0),"These filters/strides doesnt work with convolution layers"
        
        
        """  Load the training data """
        x_data,y_data=thr.loadChars74k(path+"\\train",num_classes,num_samples)
        
        """  Load valuation data  """
        x_data_eval,y_data_eval=thr.loadChars74k(path+"\\validation",num_classes,num_samples)
               
        """  Load testing data  """
        x_data_test,y_data_test=thr.loadChars74k(path+"\\test",num_classes,num_samples)

        """ tTensorflow placeholders for input    """
        x=tf.placeholder("float32", [None,50,50,3], name="x")
        y=tf.placeholder("float32", [None,num_classes], name="y")
        learning_rate=tf.placeholder("float32", (), name="learning_rate")
        drop_rate=tf.placeholder("float32", (), name="drop_rate")
        
        """  Adding convolutional layers         """
        conv1_w1=create_weights(filters[0])
        conv1=tf.nn.conv2d(x,filter=conv1_w1,strides=strides[0],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',    name="COnv1")
        conv1_relu=tf.nn.relu(conv1,name="relu_conv1")
        conv2_w2=create_weights(filters[1])
        conv2=tf.nn.conv2d(conv1_relu,filter=conv2_w2,strides=strides[1],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',    name="COnv2")
        conv2_relu=tf.nn.relu(conv2,name="relu_conv2")
        conv3_w3=create_weights(filters[2])
        conv3=tf.nn.conv2d(conv2_relu,filter=conv3_w3,strides=strides[2],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',  name="COnv3")
        conv3_relu=tf.nn.relu(conv3,name="relu_conv3")
        conv4_w4=create_weights(filters[3])
        conv4=tf.nn.conv2d(conv3_relu,filter=conv4_w4,strides=strides[3],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',   name="COnv4")
        conv4_relu=tf.nn.relu(conv4,name="relu_conv4")
        
        
        """ Flattening out to pass it to fully connected layers   """
        
        conv4_flat=tf.reshape(tensor=conv4_relu, shape=[-1,sh[6]*sh[7]*filters[3][3]], name="conv4_flt")
        print(conv4_flat)
        wfc1=create_weights((sh[6]*sh[7]*filters[3][3],int(fcl[0])))
        
        fc1=tf.matmul(conv4_flat, wfc1, name="fc1")
        fc1_relu=tf.nn.relu(fc1,"relu_fc1")
        fc1_drop=tf.nn.dropout(fc1_relu, keep_prob=drop_rate, seed=10, name="dropout_fc1")
        wfc2=create_weights((int(fcl[0]),num_classes))
        logits=tf.matmul(fc1_drop, wfc2, name="fc2")
        
        softmax=tf.nn.softmax(logits)
        c_matrix=tf.confusion_matrix(tf.argmax(y,axis=1),tf.argmax(softmax,axis=1))
        
        
        """   Calculate the loss          """
        loss=tf.nn.softmax_cross_entropy_with_logits(None,labels=y, logits=logits, dim=-1, name="loss")
        
        cost=tf.reduce_mean(loss)
        
        """  Use adam optimizer   """
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        """   calculating accuracy   """
        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=1), tf.argmax(y,axis=1)),dtype=tf.float32))
        neg_count=tf.count_nonzero(tf.cast(~tf.equal(tf.argmax(logits,axis=1), tf.argmax(y,axis=1)),dtype=tf.float32))
        pos_count=tf.count_nonzero(tf.cast(tf.equal(tf.argmax(logits,axis=1), tf.argmax(y,axis=1)),dtype=tf.float32))
        
        n_samples=len(x_data)
        idx=np.arange(n_samples)
        np.random.shuffle(idx)
        init_op=tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        
        """
        This below block actually starts the training, till now we have built only computational graph
        
        """
        
        with tf.Session() as sess:
            sess.run(init_op)
            for e in range(EPOCHS):
                start_idx=0
                for s in range(math.ceil(n_samples/BATCH_SIZE)):
                    
                    if s==0:
                        x_batch,y_batch=x_data[idx[start_idx:start_idx+BATCH_SIZE]],y_data[idx[start_idx:start_idx+BATCH_SIZE]]
                        _,s_conv1,s_cost,train_accu,s_conv1_w1,s_conv2_w2,s_conv3_w3,s_conv4_w4=sess.run([optimizer,conv1,cost,accuracy,conv1_w1,conv2_w2,conv3_w3,conv4_w4],feed_dict={x:x_batch,y:y_batch,learning_rate:lr,drop_rate:dropout_rate})
                        start_idx += BATCH_SIZE
                        
                        """ This below block used for visualising the activation map   """
                        temp_img=[misc.imresize(s_conv1[i].mean(axis=2),size=(50,50),interp='bicubic') for i in range(5)]        
                        imgs=[ImageTk.PhotoImage(image=Image.fromarray(i)) for i in temp_img]
                        del temp_img
                        del s_conv1
                        
                        for l,img in zip(imglabels,imgs):
                            l.config(image=img)
                        
                        
                        """  Below block used for visualising weights as they get trained   """
                        
                        temp_img = [misc.imresize(s_conv1_w1[:,:,0:3,i],size=(50,50,3),interp='lanczos') for i in range(3)]
                        temp_img += [misc.imresize(s_conv2_w2[:,:,0:3,i],size=(50,50,3),interp='lanczos') for i in range(3)]
                        temp_img += [misc.imresize(s_conv3_w3[:,:,0:3,i],size=(50,50,3),interp='lanczos') for i in range(3)]
                        temp_img += [misc.imresize(s_conv4_w4[:,:,0:3,i],size=(50,50,3),interp='lanczos') for i in range(3)]        
                        weightimgs=[ImageTk.PhotoImage(image=Image.fromarray(i).convert("RGB")) for i in temp_img]
                                            
                        #del temp_img
                        del s_conv1_w1
                        del s_conv2_w2
                        del s_conv3_w3
                        del s_conv4_w4
                        
                        for wl,wimg in zip(weightLabels,weightimgs):
                            wl.config(image=wimg)

    
                    else:
                        x_batch,y_batch=x_data[idx[start_idx:start_idx+BATCH_SIZE]],y_data[idx[start_idx:start_idx+BATCH_SIZE]]
                        _,log,s_cost,train_accu=sess.run([optimizer,logits,cost,accuracy],feed_dict={x:x_batch,y:y_batch,learning_rate:lr,drop_rate:dropout_rate})
                        start_idx += BATCH_SIZE
                                        
                s_cost,val_accu=sess.run([cost,accuracy],feed_dict={x:x_data_eval,y:y_data_eval,learning_rate:lr,drop_rate:1.0})
                
                """   Update the validation results   """
                
                epoch_label.config(text="EPOCHS: "+str(e))
                steps_label.config(text="STEPS: "+str(s))
                train_accuracy_label.config(text="TRAIN ACCU: "+str(round(train_accu*100,2)))
                val_accuracy_label.config(text="VALIDATION ACCU: "+str(round(val_accu*100,2)))
                test_accuracy_label.config(text="TEST ACCU: "+str(round(test_accu*100,2)))
            
            
            """   update the testing results   """
            s_cost,test_accu,c_m,p_c,n_c=sess.run([cost,accuracy,c_matrix,pos_count,neg_count],feed_dict={x:x_data_test,y:y_data_test,learning_rate:lr,drop_rate:1.0})
            
            print("pos_count",p_c)
            print("neg_count",n_c)
            print("Confusion matrix\n",c_m)

            
            epoch_label.config(text="EPOCHS: "+str(e))
            steps_label.config(text="STEPS: "+str(s))
            train_accuracy_label.config(text="TRAIN ACCU: "+str(round(train_accu*100,2)))
            val_accuracy_label.config(text="VALIDATION ACCU: "+str(round(val_accu*100,2)))
            test_accuracy_label.config(text="TEST ACCU: "+str(round(test_accu*100,2)))
            

            """   Save the model   """            
            model_file=path+"\\tf_10E_400S_4L_SoftCrossEnt_normalisedV1_10C.model"
            saver.save(sess, save_path=model_file)
            
            """ Save the model in a way our app can understand  """
            
            model_config=path+"\\tf_10E_400S_4L_SoftCrossEnt_normalisedV1_10C.model_config"
            with open(model_config,'w') as f:
                json.dump({"filters":filters,"strides":strides,"lr":lr,"dropout_rate":dropout_rate,"padding":padding,"fcl":fcl,"num_classes":num_classes,"num_samples":num_samples,"model_file":model_file},f)

    finally:
        pass


if __name__=="__main__":
    train()
    
    
    