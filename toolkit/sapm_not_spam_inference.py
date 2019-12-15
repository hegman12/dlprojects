import tensorflow as tf
import numpy as np
import json

def run(s,lbl,meta_file):
    tf.set_random_seed(10)
    
    print("inside")
    
    with open("D:\\Deep Learning\\data\\practice\\uci\\YouTube-Spam-Collection-v1\\char_to_idx.json",'r') as f:
        dictionary=json.load(f)
    
    data=list(s.strip())
    print(data)
    x_list=[]
    """
    for words in enumerate(data):
        try:            
            x_list.append(dictionary[words])
        except KeyError:
            x_list.append(0)
    """
    
    data=[a if a in data else 'UNK' for a in data if a!='UNK' or a!='\n']
    
    x_list=[dictionary[a] for a in data]
    
    x_list=np.eye(315)[x_list]
    print(x_list.shape)
    
    x_len=len(x_list)
    
    tf.reset_default_graph()
    ns = tf.train.import_meta_graph(meta_file , clear_devices=True)
    with tf.Session().as_default() as sess:
        ns.restore(sess, meta_file[0:len(meta_file)-5])
        x=tf.get_default_graph().get_tensor_by_name("x:0")
        lr=tf.get_default_graph().get_tensor_by_name("lr:0")
        cell_state=tf.get_default_graph().get_tensor_by_name("cell_state:0")
        hidden_state=tf.get_default_graph().get_tensor_by_name("hidden_state:0")
        
        #xx=np.array(x_list,dtype=np.int)

        cs=np.zeros((3,1,315))
        hs=np.zeros((3,1,315))
                
        op=tf.get_default_graph().get_tensor_by_name("final_out:0")        
        o=sess.run(op,feed_dict={x:x_list[np.newaxis,:,:],lr:0.001,hidden_state:hs,cell_state:cs})
        #o=sess.run(op1,feed_dict={x:xx,lr:0.001,cell_state:cs,hidden_state:hs})
        soft_out=tf.nn.softmax(o)
        result=np.argmax(soft_out,axis=-1)
        lbl.config(text="Spam! "+ str(soft_out.eval()) if result==0 else "Not Spam! " + str(soft_out.eval()))
        sess.close()