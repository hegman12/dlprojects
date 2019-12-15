import tensorflow as tf
import numpy as np
import json

def run(s,lbl,meta_file):
    tf.set_random_seed(10)
    
    with open("D:\\Deep Learning\\model\\RNN\\sentiment\\dict.json.txt",'r') as f:
        dictionary=json.load(f)
    
    data=s.replace('.',' ').split()
    x_list=[]
    """
    for words in enumerate(data):
        try:            
            x_list.append(dictionary[words])
        except KeyError:
            x_list.append(0)
    """
    
    data=[a if a in dictionary else 'UNK' for a in data]
    
    x_list=[dictionary[a] for a in data]
    
    x_len=len(x_list)
    
    tf.reset_default_graph()
    ns = tf.train.import_meta_graph(meta_file , clear_devices=True)
    with tf.Session().as_default() as sess:
        ns.restore(sess, meta_file[0:len(meta_file)-5])
        x=tf.get_default_graph().get_tensor_by_name("x:0")
        lr=tf.get_default_graph().get_tensor_by_name("lr:0")
        bs=tf.get_default_graph().get_tensor_by_name("batch_size:0")
        output_keep_prob=tf.get_default_graph().get_tensor_by_name("output_keep_prob:0")
        
        xx=np.array(x_list,dtype=np.int)
                
        op=tf.get_default_graph().get_tensor_by_name("soft_out:0")        
        o=sess.run(op,feed_dict={x:xx.reshape(1,x_len),lr:0.001,bs:1,output_keep_prob:1.0})
        #o=sess.run(op1,feed_dict={x:xx,lr:0.001,cell_state:cs,hidden_state:hs})
        print(o)
        result=np.argmax(o,axis=-1)
        lbl.config(text="Negetive"+ str(o) if result==0 else "Positive" + str(o))
        sess.close()