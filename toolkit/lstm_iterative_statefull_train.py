import tensorflow as tf
import numpy as np
import datasets
import matplotlib.pyplot as plt

#filename,lrv,drv,epoc,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label

def lstm_cell(state_size):
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(state_size))
batch_size=50
def graph(char_to_idx_len,num_layers):
    x=tf.placeholder(tf.float32,(None,None,char_to_idx_len),"x")
    y=tf.placeholder(tf.int32,(),"y")
    lr=tf.placeholder(tf.float32,(),"lr")
    cell_state=tf.placeholder(tf.float32,(num_layers,1,char_to_idx_len),"cell_state")
    hidden_state=tf.placeholder(tf.float32,(num_layers,1,char_to_idx_len),"hidden_state")
    state=tf.placeholder(tf.float32,(num_layers,2,1,char_to_idx_len),"state")
    w=tf.Variable(initial_value=tf.truncated_normal(shape=(char_to_idx_len,char_to_idx_len), mean=0, stddev=1, dtype="float32", seed=10, name="w"))
    b=tf.Variable(initial_value=tf.truncated_normal(shape=(char_to_idx_len,), mean=0, stddev=1, dtype="float32", seed=10, name="b"))
    init_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(c,h) for (c,h) in zip(tf.unstack(cell_state),tf.unstack(hidden_state)))
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(char_to_idx_len) for _ in range(num_layers)])
    out,state=tf.nn.dynamic_rnn(cell,x,initial_state=init_state)
    print(out)
    state_tensor=tf.identity(state,name="state_tensor")
    h=tf.reshape(out,[-1,char_to_idx_len])
    out=tf.add(tf.matmul(h,w),b)
    final_out=tf.identity(out[-1],name="final_out")
    
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=final_out)
    train_step=tf.train.AdagradOptimizer(lr).minimize(loss)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,tf.cast(tf.argmax(final_out),tf.int32)),tf.float32))
    return x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state


def train(filename,lrv,drv,epoc,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
    
    data,indexes=datasets.spam_notspam_youtube_rnn_classification(True,filename)
    
    x_train,y_train,x_test,y_test=data
    
    unique_chars,char_to_idx,idx_to_chr,max_len=indexes
    tf.set_random_seed(10)
    
    char_to_idx_len=len(char_to_idx)
    
    tf.reset_default_graph()
    num_layers=3    

    x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state=graph(char_to_idx_len,num_layers)
     
    init=tf.global_variables_initializer()
    
    data_size=len(x_train)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    voc_size=len(char_to_idx)
    counter=0
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        cs=np.zeros((3,1,315))
        hs=np.zeros((3,1,315))
        loss_counter=list()
        for e in range(epoc):
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
                counter += a
                loss_counter.append(l)
                if s%20==0:
                    print("Iteration",s,"Loss",np.mean(np.array(loss_counter)),"Accuracy",counter/20)
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    train_accuracy_label.config(text="TRAIN ACCU: "+str(counter/20))
                    #val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")
                    counter=0
                    loss_counter=[]
                if s%500==0:
                    #val_cs=np.zeros((1,315))
                    #val_hs=np.zeros((1,315))
                    val_cs=np.copy(cs)
                    val_hs=np.copy(hs)
                    val_counter=0
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
                        val_counter += va
                        val_loss_counter.append(vl)
                    print("Val Loss",np.mean(np.array(val_loss_counter)),"val Accuracy",val_counter/1535)
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    #train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    val_accuracy_label.config(text="VALIDATION ACCU: "+str(val_counter/1535))
                    test_accuracy_label.config(text="TEST ACCU: NA")
                    
        np.save("D:\\Deep Learning\\model\\RNN\\sentiment\\state",st)
        saver.save(sess,"D:\\Deep Learning\\model\\RNN\\sentiment\\spam_not_spam")

if __name__=="__main__":
    train()
