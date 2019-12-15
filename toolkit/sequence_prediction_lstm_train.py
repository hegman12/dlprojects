import numpy as np
import tensorflow as tf


def load_data(path):
    files=[ str(a-1)+"-"+str(a) for a in range(1997,2019,1)]
    data=np.empty((1,4))
    
    def converter(s):
        #print(s.decode("utf-8").strip("\"").strip())
        return float(s.decode("utf-8").strip("\"").strip())

    for file in files:
        temp_data=np.genfromtxt(fname=path+"\\"+file+".csv",dtype=np.float,skip_header=1,delimiter=",",usecols=(1,2,3,4),converters={1:converter,2:converter,3:converter,4:converter})
        data=np.concatenate((data,temp_data),axis=0)
    
    return data[:,3]
  
np.random.seed(10)
seq_size=25
state_size=64
out_size=1
#keep_prob=0.4
reg_scale=0.1

def lstm_cell(state_size,keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(state_size),output_keep_prob=keep_prob)

def graph():
    tf.set_random_seed(10)
    tf.reset_default_graph()
    weight_initializer=tf.truncated_normal_initializer()
    bias_init=tf.zeros_initializer()
    
    x=tf.placeholder(tf.float32,shape=(1,seq_size,1),name="x")
    y=tf.placeholder(tf.float32,shape=(1,1),name="y")
    keep_prob=tf.placeholder(tf.float32,shape=(),name="keep_prob")
    lr=tf.placeholder(tf.float32,shape=(),name="lr")

    reg_scale=tf.constant(dtype=tf.float32,value=0.01)
    
    w=tf.get_variable("w",shape=(state_size,out_size),initializer=weight_initializer)
    b=tf.get_variable("b",shape=(),initializer=bias_init)
    
    cell_state=tf.placeholder(tf.float32,shape=(1,state_size),name="cell_state")
    hidden_state=tf.placeholder(tf.float32,shape=(1,state_size),name="hidden_state")
    
    init_state= tuple(tf.nn.rnn_cell.LSTMStateTuple(tf.reshape(c,[1,-1],name="c_reshape"),tf.reshape(s,[1,-1],name="s_reshape")) for (c,s) in zip(tf.unstack(cell_state),tf.unstack(hidden_state)))

    cell=lstm_cell(state_size,keep_prob)
    rnn_output,state=tf.nn.dynamic_rnn(cell,x,initial_state=init_state[0])
    
    rnn_drop=tf.nn.dropout(rnn_output[:,-1,:],keep_prob=1.0,name="rnn_drop")
    
    output= tf.add(tf.matmul(rnn_drop,w,name="matmul"),b,name="add")
        
    loss=tf.multiply(tf.add(tf.losses.mean_squared_error(labels=y,predictions=output),tf.nn.l2_loss(w),name="loss"),reg_scale,name="scaled_loss")
    #loss,update_op=tf.metrics.root_mean_squared_error(labels=y,predictions=output)
    #print(loss)
    optimizer=tf.train.GradientDescentOptimizer(0.001)
    
    train_step=optimizer.minimize(loss)
    
    return train_step,x,y,loss,state,cell_state,hidden_state,output,keep_prob,lr

def r_squared(actual_value,predicted_value):
    av=np.array(actual_value)
    pv=np.array(predicted_value)
    
    total_error= np.mean(np.square(np.subtract(av,np.mean(av))))
    unexplained_error=np.mean(np.subtract(av,pv))
    rsquare=np.subtract(np.divide(unexplained_error,total_error),1)
    print("Goodness of fit",rsquare)

def train(path,lrv,drv,epocv,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label):
    
    #"D:\\Deep Learning\\data\\practice\\NSE"
    
    data=load_data(path)
    train=data[0:4500]
    test=data[4500-10:None]
    
    train_step,x,y,loss,state,cell_state,hidden_state,output,keep_prob,lr=graph()
    init_op=tf.global_variables_initializer()
    local_op=tf.local_variables_initializer()
    saver=tf.train.Saver()
    
    counter=list()
    batch_size=1
    actual_value=[]
    predicted_value=[]
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(local_op)
        loss_counter=list()
        cs=np.zeros((1,state_size))
        hs=np.zeros((1,state_size))
        for e in range(epocv):
            counter=[]
            loss_counter=[]
    
            for s in range(0,4500-seq_size-1,batch_size):
                xx=train[s:s+seq_size].reshape(1,-1)
                #xx=xx/np.max(xx)-np.min(xx)
                #print(xx)
                y_list=train[s+seq_size].reshape(1,-1)
                y_list=np.array(y_list,np.float32)
                fd={x:xx[:,:,np.newaxis],y:y_list,cell_state:cs,hidden_state:hs,keep_prob:drv,lr:lrv}
                l,_,st=sess.run([loss,train_step,state],feed_dict=fd)
    
                cs[0]=st.c
                hs[0]=st.h
                #counter.append(a)
                loss_counter.append(l)
                
                if s%500==0:
                    print("Epoch",e,"Loss",np.mean(np.array(loss_counter)))
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    #val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")

                    counter=[]
                    loss_counter=[]
                
                if s%1500==0:
                    #val_cs=np.zeros((1,state_size))
                    #val_hs=np.zeros((1,state_size))
                    val_counter=[]
                    val_loss_counter=[]
                    val_cs=np.zeros_like(cs)
                    val_hs=np.zeros_like(hs)
                    for v in range(0,976-seq_size-1,batch_size):
                        val_xx=test[v:v+seq_size].reshape(1,-1)
                        #val_xx=val_xx/np.max(val_xx)-np.min(val_xx)
                        val_yy=test[v+seq_size].reshape(1,-1)
                        vfd={x:val_xx[:,:,np.newaxis],y:np.array(val_yy,np.float32),cell_state:val_cs,hidden_state:val_hs,keep_prob:1.0,lr:lrv}
                        vl,val_state,out=sess.run([loss,state,output],feed_dict=vfd)
                        val_cs[0]=val_state.c
                        #val_cs[1]=val_state[1].c
                        #val_cs[2]=val_state[2].c
                        val_hs[0]=val_state.h
                        #val_hs[1]=val_state[1].h
                        #val_hs[2]=val_state[2].h
                        #val_counter.append(va)
                        val_loss_counter.append(vl)
                        actual_value.append(val_yy[0][0])
                        predicted_value.append(out[0][0])
                    r_squared(actual_value,predicted_value)
                    actual_value=[]
                    predicted_value=[]
                    print("Epoch",e,"Val Loss",np.mean(np.array(val_loss_counter)))
                    epoch_label.config(text="EPOCHS: "+str(e))
                    steps_label.config(text="STEPS: "+str(s))
                    #train_accuracy_label.config(text="TRAIN ACCU: "+str(np.mean(counter)))
                    val_accuracy_label.config(text="VALIDATION ACCU: "+str(np.mean(val_counter)))
                    test_accuracy_label.config(text="TEST ACCU: NA")
            #print("Epoch",e,"Loss",np.mean(np.array(loss_counter)),"Accuracy",np.mean(counter))
        print("Val Loss",np.mean(np.array(val_loss_counter)))
        #np.save("D:\\Deep Learning\\model\\RNN\\sentiment\\senti_state_batch",st)
        saver.save(sess,"D:\\Deep Learning\\model\\RNN\\NSE\\model")
    
if __name__=="__main__"  :
    train("D:\\Deep Learning\\data\\practice\\NSE")


















       
    