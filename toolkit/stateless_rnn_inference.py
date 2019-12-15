import tensorflow as tf
import numpy as np
import datasets
import utils_rnn


def run(s,lbl,meta_file):

    data,indexes=datasets.spam_notspam_youtube_rnn_classification(True)
    
    unique_chars,char_to_idx,idx_to_chr,max_len=indexes
    
    tf.set_random_seed(10)
    
    l=[]
    for word in s.split():
        for c in word:    
            l.append(char_to_idx[c])
    tf.set_random_seed(10)
    char_to_idx['UNKNOWN']=-1
    idx_to_chr[-1]='UNKNOWN'
    xx=utils_rnn.pad_data([l],max_len,value_to_fill_int=-1) 
    
    tf.reset_default_graph()
    ns = tf.train.import_meta_graph(meta_file , clear_devices=True)
    with tf.Session().as_default() as sess:
        ns.restore(sess, meta_file[0:len(meta_file)-5])
        x=tf.get_default_graph().get_tensor_by_name("x:0")
        lr=tf.get_default_graph().get_tensor_by_name("lr:0")
        cell_state=tf.get_default_graph().get_tensor_by_name("cell_state:0")
        bs=tf.get_default_graph().get_tensor_by_name("batch_size:0")
        output_keep_prob=tf.get_default_graph().get_tensor_by_name("output_keep_prob:0")
        #w=tf.get_default_graph().get_tensor_by_name("w:0")
        hidden_state=tf.get_default_graph().get_tensor_by_name("hidden_state:0")
        cs=np.zeros((3,1,316))
        hs=np.zeros((3,1,316))
    
        op=tf.get_default_graph().get_tensor_by_name("final_out:0")
        op1=tf.nn.softmax(op)
        o=sess.run(op1,feed_dict={x:xx,lr:0.001,cell_state:cs,hidden_state:hs,bs:1,output_keep_prob:1.0})
        #o=sess.run(op1,feed_dict={x:xx,lr:0.001,cell_state:cs,hidden_state:hs})
        result=np.argmax(o,axis=-1)[0]
        lbl.config(text="Not Spam!" if result==0 else "Spam!" + str(o))
        sess.close()