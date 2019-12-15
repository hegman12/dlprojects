import collections
import numpy as np
import random
import tensorflow as tf
import math
import utils_rnn


def load_data(path):
    x_data=[]
    y_data=[]
    with open(path+"//yelp_labelled.txt") as f:
        for line in f:
            x_data.append(line.split('\t')[0])
            y_data.append(int(line.split('\t')[1].strip()))
    
    with open(path+"//imdb_labelled.txt") as f:
        for line in f:
            x_data.append(line.split('\t')[0])
            y_data.append(int(line.split('\t')[1].strip()))
    
    with open(path+"//amazon_cells_labelled.txt") as f:
        for line in f:
            x_data.append(line.split('\t')[0])
            y_data.append(int(line.split('\t')[1].strip()))
    
    return x_data,y_data


def get_numpy_emb_mat():
    em=np.load('D:\\Deep Learning\\model\\embeddings\\final_embeddings_norm.npy',mmap_mode='r')
    return em


def lstm_cell(state_size,output_keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(state_size),output_keep_prob=output_keep_prob)
    #return tf.nn.rnn_cell.LSTMCell(state_size)
def graph(em,batch_size,max_len):
    tf.set_random_seed(10)
    tf.reset_default_graph()
    num_layers=1
    out_unit=2
    state_size=8
    embedding_size=em.shape[-1]
    b_size=tf.placeholder(tf.int32,shape=(),name="batch_size")
    x=tf.placeholder(tf.int32,(None,max_len),"x")
    embedding_matrix=tf.constant(em)
    x_flat=tf.reshape(x,[-1])
    embeddings=tf.nn.embedding_lookup(embedding_matrix,x_flat)
    xo=tf.reshape(embeddings,[b_size,max_len,embedding_size])    
    xo=tf.identity(xo,name="x_onehot")
    #print(xo[:,1,:])
    y=tf.placeholder(tf.int32,(None,),"y")
    lr=tf.placeholder(tf.float32,(),"lr")
    output_keep_prob=tf.placeholder(tf.float32,name="output_keep_prob")
    #cell_state=tf.placeholder(tf.float32,(num_layers,None,state_size),"cell_state")
    #hidden_state=tf.placeholder(tf.float32,(num_layers,None,state_size),"hidden_state")
    #state=tf.placeholder(tf.float32,(num_layers,2,None,out_unit),"state")
    w=tf.Variable(initial_value=tf.truncated_normal(shape=(state_size,out_unit), mean=0, stddev=2.0/np.sqrt(state_size), dtype="float32", name="w"))
    b=tf.Variable(initial_value=tf.zeros(out_unit), name="b")
    #init_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(c,h) for (c,h) in zip(tf.unstack(cell_state),tf.unstack(hidden_state)))
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(state_size,output_keep_prob) for _ in range(num_layers)])
    #init_state=state=cell.zero_state(batch_size,dtype=tf.float32)
    #for i in range(max_len):
    #    rnn_out,state=cell(xo[:,i,:],state)
    #print(rnn_out)
    init_state=cell.zero_state(b_size,dtype=tf.float32)
    rnn_out,state=tf.nn.dynamic_rnn(cell,xo,initial_state=init_state)
    #rnn_out=rnn_out[:,-1,:]
    h=rnn_out[:,-1,:]
    print(rnn_out)
    state_tensor=tf.identity(state,name="state_tensor")
    #h1=tf.reshape(rnn_out,[-1,state_size])
    h=tf.nn.dropout(h,keep_prob=output_keep_prob)
    linear_out=tf.add(tf.matmul(h,w),b)
    print(linear_out)
    #out=tf.reshape(linear_out,[batch_size,-1,out_unit])
    #out=tf.Print(out,[tf.shape(out)],"OUT: ")
    #final_out=tf.identity(out[:,-1,:],name="final_out")
    #final_out=tf.reduce_sum(out,1,name="final_out")
    #print(final_out)
    #final_out=tf.Print(final_out,[tf.shape(final_out)],"FO: ")
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,2),logits=linear_out))
    train_step=tf.train.AdamOptimizer(lr).minimize(loss)
    soft_out=tf.nn.softmax(linear_out)
    argmax_out=tf.argmax(soft_out,axis=-1)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,tf.cast(argmax_out,tf.int32)),tf.float32))
    return x,y,lr,linear_out,loss,train_step,accu,b_size,output_keep_prob,soft_out
    
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
   
def train(path,lrv,drv,epocv,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
    x_data,y_data=load_data(path)
    batch_size=32
    data_sentiment=[[a for a in b.replace('.',' ').split()] for b in x_data]
    max_len=15
    vocabulary_size = 10000
    
    data=' '.join([' '.join(a) for a in data_sentiment])
    data, count, dictionary, reverse_dictionary = build_dataset(data.split(), vocabulary_size)
    
    x_list=[[dictionary[a] for a in b] for b in data_sentiment]
    l=[ len(a) for a in x_list]
    bv=[a<=max_len for a in l]
    x_list1=[ll for ll,bvv in zip(x_list,bv) if bvv==True]
    y_list=[ll for ll,bvv in zip(y_data,bv) if bvv==True]
    
    em=get_numpy_emb_mat()
    
    x,y,lr,final_out,loss,train_step,accu,bs,output_keep_prob,soft_out=graph(em,batch_size,max_len)
    
    init=tf.global_variables_initializer()
    
    x_train=x_list1[0:2100]
    y_train=y_list[0:2100]
    x_test=x_list1[2100:None]
    y_test=y_list[2100:None]
    
    counter=list()
    saver=tf.train.Saver()
    try:    
        with tf.Session() as sess:
            sess.run(init)
            loss_counter=list()
            for e in range(epocv):
                counter=[]
                loss_counter=[]
                for s in range(0,2100,batch_size):
                    xx=x_train[s:s+batch_size]
                    xx=utils_rnn.pad_data(data=xx,max_length=max_len,value_to_fill_int=0)
                    y_list=y_train[s:s+batch_size]
                    y_list=np.array(y_list,np.int)
                    c_batch_size=len(y_list)
                    fd={x:xx,y:y_list,lr:lrv,bs:c_batch_size,output_keep_prob:drv}
                    l,_,a=sess.run([loss,train_step,accu],feed_dict=fd)
                    counter.append(a)
                    loss_counter.append(l)
                    
                    if s%100==0:
                        epoch_label.config(text="EPOCHS: "+str(e))
                        steps_label.config(text="STEPS: "+str(s))
                        train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                        #val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                        test_accuracy_label.config(text="TEST ACCU: NA")
                    
                    
                    if s%500==0:
                        val_counter=[]
                        val_loss_counter=[]
                        for v in range(0,83,batch_size):
                            #val_cs=np.zeros_like(cs)
                            #val_hs=np.zeros_like(hs)
                            val_xx=x_test[v:v+batch_size]
                            val_xx=utils_rnn.pad_data(data=val_xx,max_length=max_len,value_to_fill_int=0)
                            val_yy=y_test[v:v+batch_size]
                            c_batch_size=len(val_yy)
                            vfd={x:val_xx,y:np.array(val_yy,np.int),lr:lrv,bs:c_batch_size,output_keep_prob:1.0}
                            vl,va=sess.run([loss,accu],feed_dict=vfd)
                            val_counter.append(va)
                            val_loss_counter.append(vl)
                            
                            epoch_label.config(text="EPOCHS: "+str(e))
                            steps_label.config(text="STEPS: "+str(s))
                            #train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                            val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                            test_accuracy_label.config(text="TEST ACCU: NA")
                    
                print("Epoch",e,"Loss",np.mean(np.array(loss_counter)),"Accuracy",np.mean(counter))
            print("Val Loss",np.mean(np.array(val_loss_counter)),"val Accuracy",np.mean(val_counter))
            saver.save(sess,"D:\\Deep Learning\\model\\RNN\\sentiment\\senti_batch")
    finally:
        del em
    
    
    
    
    
    
    
    