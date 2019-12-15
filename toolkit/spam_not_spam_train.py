import collections
import numpy as np
import random
import tensorflow as tf
import math
import datasets
import utils_rnn

global char_to_idx

def load_data():
    global char_to_idx
    
    data,indexes=datasets.spam_notspam_youtube_rnn_classification(True)
    x_train,y_train,x_test,y_test=data
    unique_chars,char_to_idx,idx_to_chr,max_len=indexes
    
    return x_train,y_train,x_test,y_test,unique_chars,char_to_idx,idx_to_chr,max_len


def lstm_cell(state_size,output_keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(state_size),output_keep_prob=output_keep_prob)
    #return tf.nn.rnn_cell.LSTMCell(state_size)


def graph(batch_size=50,num_layers=3):
    
    global char_to_idx
    
    x=tf.placeholder(tf.float32,(None,None,len(char_to_idx)),"x")
    y=tf.placeholder(tf.int32,(),"y")
    lr=tf.placeholder(tf.float32,(),"lr")
    cell_state=tf.placeholder(tf.float32,(num_layers,1,len(char_to_idx)),"cell_state")
    hidden_state=tf.placeholder(tf.float32,(num_layers,1,len(char_to_idx)),"hidden_state")
    state=tf.placeholder(tf.float32,(num_layers,2,1,len(char_to_idx)),"state")
    w=tf.Variable(initial_value=tf.truncated_normal(shape=(len(char_to_idx),2), mean=0, stddev=1, dtype="float32", seed=10, name="w"))
    b=tf.Variable(initial_value=tf.truncated_normal(shape=(len(char_to_idx),), mean=0, stddev=1, dtype="float32", seed=10, name="b"))
    init_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(c,h) for (c,h) in zip(tf.unstack(cell_state),tf.unstack(hidden_state)))
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(len(char_to_idx),0.4) for _ in range(num_layers)])
    out,state=tf.nn.dynamic_rnn(cell,x,initial_state=init_state)
    print(out)
    state_tensor=tf.identity(state,name="state_tensor")
    h=tf.reshape(out,[-1,len(char_to_idx)])
    out=tf.add(tf.matmul(h,w),b)
    final_out=tf.identity(out[-1],name="final_out")
    
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=final_out)
    train_step=tf.train.AdagradOptimizer(lr).minimize(loss)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,tf.cast(tf.argmax(final_out),tf.int32)),tf.float32))
    return x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state
    
   
#def train(path,lrv,drv,epocv,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
def train(lrv,epocv,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
    
    x_train,y_train,x_test,y_test,unique_chars,char_to_idx,idx_to_chr,max_len=load_data()
    
    x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state=graph()
    
    init=tf.global_variables_initializer()
    
    data_size=len(x_train)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    voc_size=len(char_to_idx)
    counter=[]
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        cs=np.zeros((3,1,315))
        hs=np.zeros((3,1,315))
        loss_counter=list()
        for e in range(epocv):
            for s in range(0,6000,1):
                xx=np.eye(voc_size)[x_train[s]]
                yy=y_train[s]
                fo,l,_,a,st=sess.run([final_out,loss,train_step,accu,state],feed_dict={x:xx[np.newaxis,:,:],y:yy,lr:lrv,cell_state:cs,hidden_state:hs})
                
                cs[0]=st[0].c
                cs[1]=st[1].c
                cs[2]=st[2].c
                hs[0]=st[0].h
                hs[1]=st[1].h
                hs[2]=st[2].h
                counter.append(a)
                loss_counter.append(l)
                if s%20==0:
                    print("Iteration",s,"Loss",np.mean(np.array(loss_counter)),"Accuracy",str(np.mean(counter)))
                    
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    #val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")
                    counter=[]
                    loss_counter=[]
                    
                if s%500==0:
                    #val_cs=np.zeros((1,315))
                    #val_hs=np.zeros((1,315))
                    val_cs=np.copy(cs)
                    val_hs=np.copy(hs)
                    val_counter=[]
                    val_loss_counter=[]
                    for v in range(1535):
                        val_xx=np.eye(voc_size)[x_test[v]]
                        val_yy=y_test[v]
                        vl,va,val_state=sess.run([loss,accu,state],feed_dict={x:val_xx[np.newaxis,:,:],y:val_yy,lr:0.001,cell_state:val_cs,hidden_state:val_hs})
                        val_cs[0]=val_state[0].c
                        val_cs[1]=val_state[1].c
                        val_cs[2]=val_state[2].c
                        val_hs[0]=val_state[0].h
                        val_hs[1]=val_state[1].h
                        val_hs[2]=val_state[2].h
                        val_counter.append( va)
                        val_loss_counter.append(vl)
                    print("Val Loss",np.mean(np.array(val_loss_counter)),"val Accuracy",str(np.mean(val_counter)))
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    #train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")
        np.save("D:\\Deep Learning\\model\\RNN\\sentiment\\state",st)
        saver.save(sess,"D:\\Deep Learning\\model\\RNN\\sentiment\\spam_not_spam")

if __name__=="__main__":
    train()
    