import tensorflow as tf
import numpy as np
import utils_rnn

def load(data_path):
    x=[]
    y=[]
    unique_chars=set()
    max_len=0
    
    FILE=data_path+"\\yelp_labelled.txt"
    with open(FILE,"r", encoding="utf8") as f:
        for line in f:
            temp=line.split("\t")[0].strip()
            x.append(list(temp))
            y.append(int(line.split("\t")[1].strip()))
            unique_chars=set(''.join(unique_chars)+temp)
            
            if len(temp)>max_len:
                max_len=len(temp)
 
    FILE=data_path+"\\imdb_labelled.txt"
    with open(FILE,"r", encoding="utf8") as f:
        for line in f:
            temp=line.split("\t")[0].strip()
            x.append(list(temp))
            y.append(int(line.split("\t")[1].strip()))
            unique_chars=set(''.join(unique_chars)+temp)
            
            if len(temp)>max_len:
                max_len=len(temp)
                
    FILE=data_path+"\\amazon_cells_labelled.txt"
    with open(FILE,"r", encoding="utf8") as f:
        for line in f:
            temp=line.split("\t")[0].strip()
            x.append(list(temp))
            y.append(int(line.split("\t")[1].strip()))
            unique_chars=set(''.join(unique_chars)+temp)
            
            if len(temp)>max_len:
                max_len=len(temp)
            
    return x,y,set(''.join(unique_chars)),max_len,len(unique_chars)


def get_data(data_path):
    x,y,unique_chars,max_len,voc_size=load(data_path)
    
    chr_to_idx={c:i for (i,c) in enumerate(unique_chars)}
    x=[ [chr_to_idx[b] for b in a] for a in x]
    x=utils_rnn.pad_data(data=x,max_length=max_len,value_to_fill_int=-1)
    
    y=np.array(y)
    x_train=x[0:2500]
    y_train=y[0:2500]
    
    x_test=x[2500:None]
    y_test=y[2500:None]
    
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    print("x_test",x_test.shape)
    print("y_test",y_test.shape)
    print("max_len",max_len)
    print("voc_size",voc_size)
    
    return x_train,y_train,x_test,y_test,max_len,voc_size
    

def lstm_cell(state_size,output_keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(state_size),output_keep_prob=output_keep_prob)
def graph(voc_size,num_layers,out_unit,state_size):
    x=tf.placeholder(tf.int32,(None,None),"x")
    xo=tf.one_hot(x,voc_size,name="x_onehot")
    y=tf.placeholder(tf.int32,(None,),"y")
    lr=tf.placeholder(tf.float32,(),"lr")
    batch_size=tf.placeholder(tf.int32,name="batch_size")
    output_keep_prob=tf.placeholder(tf.float32,name="output_keep_prob")
    cell_state=tf.placeholder(tf.float32,(num_layers,None,state_size),"cell_state")
    hidden_state=tf.placeholder(tf.float32,(num_layers,None,state_size),"hidden_state")
    state=tf.placeholder(tf.float32,(num_layers,2,None,out_unit),"state")
    w=tf.Variable(initial_value=tf.truncated_normal(shape=(state_size,out_unit), mean=0, stddev=0.1, dtype="float32", name="w"))
    b=tf.Variable(initial_value=tf.truncated_normal(shape=(out_unit,), mean=0, stddev=0.1, dtype="float32", name="b"))
    init_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(c,h) for (c,h) in zip(tf.unstack(cell_state),tf.unstack(hidden_state)))
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(state_size,output_keep_prob) for _ in range(num_layers)])
    rnn_out,state=tf.nn.dynamic_rnn(cell,xo,initial_state=init_state)
    state_tensor=tf.identity(state,name="state_tensor")
    h1=tf.reshape(rnn_out,[-1,state_size])
    h=tf.nn.dropout(h1,keep_prob=output_keep_prob)
    linear_out=tf.add(tf.matmul(h,w),b)
    
    out=tf.reshape(linear_out,[batch_size,-1,out_unit])
    out=tf.Print(out,[tf.shape(out)],"OUT: ")
    #final_out=tf.identity(out[:,-1,:],name="final_out")
    final_out=tf.reduce_sum(out,1,name="final_out")
    print(final_out)
    final_out=tf.Print(final_out,[tf.shape(final_out)],"FO: ")
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=final_out)
    train_step=tf.train.AdagradOptimizer(lr).minimize(loss)
    soft_out=tf.nn.softmax(final_out)
    argmax_out=tf.argmax(soft_out,axis=-1)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,tf.cast(argmax_out,tf.int32)),tf.float32))
    return x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state,batch_size,output_keep_prob


def train(data_path,lrv,drv,epocv,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
    np.random.seed(90)
    x_train,y_train,x_test,y_test,max_len,voc_size=get_data(data_path)
    
    tf.reset_default_graph()
    num_layers=3
    out_unit=2
    state_size=voc_size
    
    x,y,lr,final_out,loss,train_step,accu,state,cell_state,hidden_state,bs,output_keep_prob=graph(voc_size,num_layers,out_unit,state_size)
    init=tf.global_variables_initializer()    

    counter=list()
    batch_size=500
    saver=tf.train.Saver()
    
    
    with tf.Session() as sess:
        sess.run(init)
        loss_counter=list()
        val_counter=0
        for e in range(epocv):
            for s in range(0,2500,batch_size):
                cs=np.zeros((3,batch_size,voc_size))
                hs=np.zeros((3,batch_size,voc_size))
                xx=x_train[s:s+batch_size]
                yy=y_train[s:s+batch_size]
                c_batch_size=len(yy)
                fo,l,_,a,st=sess.run([final_out,loss,train_step,accu,state],feed_dict={x:xx,y:yy,lr:lrv,cell_state:cs,hidden_state:hs,bs:c_batch_size,output_keep_prob:drv})
                cs[0]=st[0].c
                cs[1]=st[1].c
                cs[2]=st[2].c
                hs[0]=st[0].h
                hs[1]=st[1].h
                hs[2]=st[2].h
                counter.append(a)
                loss_counter.append(l)
                if s%100==0:
                    print("Iteration",s,"Loss",np.mean(np.array(loss_counter)),"Accuracy",np.mean(counter))
                    
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
                    val_counter=[]
                    val_loss_counter=[]
                    for v in range(0,500,batch_size):
                        val_cs=np.zeros_like(cs)
                        val_hs=np.zeros_like(hs)
                        val_xx=x_test[v:v+batch_size]
                        val_yy=y_test[v:v+batch_size]
                        c_batch_size=len(val_yy)
                        vl,va,val_state=sess.run([loss,accu,state],feed_dict={x:val_xx,y:val_yy,lr:lrv,cell_state:val_cs,hidden_state:val_hs,bs:batch_size,output_keep_prob:1.0})
                        val_cs[0]=val_state[0].c
                        val_cs[1]=val_state[1].c
                        val_cs[2]=val_state[2].c
                        val_hs[0]=val_state[0].h
                        val_hs[1]=val_state[1].h
                        val_hs[2]=val_state[2].h
                        val_counter.append(va)
                        val_loss_counter.append(vl)
                    print("Val Loss",np.mean(np.array(val_loss_counter)),"val Accuracy",np.mean(val_counter))
                    
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    #train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")
                    
        np.save("D:\\Deep Learning\\model\\RNN\\sentiment\\senti_state_batch",st)
        saver.save(sess,"D:\\Deep Learning\\model\\RNN\\sentiment\\senti_batch")
    
if __name__=="__main__":
    train()
    
