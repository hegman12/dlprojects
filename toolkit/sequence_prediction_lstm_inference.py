import tensorflow as tf
import numpy as np

def run(s,lbl,meta_file):
    tf.set_random_seed(10)
    state_size=64
    x_list=s.strip().split()
    
    tf.reset_default_graph()
    ns = tf.train.import_meta_graph(meta_file , clear_devices=True)
    with tf.Session().as_default() as sess:
        ns.restore(sess, meta_file[0:len(meta_file)-5])
        x=tf.get_default_graph().get_tensor_by_name("x:0")
        #lr=tf.get_default_graph().get_tensor_by_name("lr:0")
        #keep_prob=tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        cell_state=tf.get_default_graph().get_tensor_by_name("cell_state:0")
        hidden_state=tf.get_default_graph().get_tensor_by_name("hidden_state:0")
        
        cs=np.zeros((1,state_size))
        hs=np.zeros((1,state_size))
        
        xx=np.array(x_list,dtype=np.int)
                
        op=tf.get_default_graph().get_tensor_by_name("output:0")        

        #o=sess.run(op,feed_dict={x:xx,lr:0.001,cell_state:cs,dr:1.0,hidden_state:hs})
        o=sess.run(op,feed_dict={x:xx.reshape(1,25,1),cell_state:cs,hidden_state:hs})
        
        print(o)
        lbl.config(text="Predicted: "+str(1.0/o))
        sess.close()