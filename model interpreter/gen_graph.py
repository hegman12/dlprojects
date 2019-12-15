import tensorflow as tf

def build_graph(strides,filters,padding,sh,fcl,drop_rate,num_classes,model_file):
        
    new_saver = tf.train.import_meta_graph(model_file + '.meta' , clear_devices=True)
    x=tf.get_default_graph().get_tensor_by_name("x:0")
    conv1_w1=tf.get_default_graph().get_tensor_by_name("Variable:0")    
    #conv1_w1=create_weights(filters[0])
    conv1=tf.nn.conv2d(x,filter=conv1_w1,strides=strides[0],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',    name="COnv1")
    conv1_relu=tf.nn.relu(conv1,name="relu_conv1")
    #conv2_w2=create_weights(filters[1])
    conv2_w2=tf.get_default_graph().get_tensor_by_name("Variable_1:0")  
    conv2=tf.nn.conv2d(conv1_relu,filter=conv2_w2,strides=strides[1],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',    name="COnv2")
    conv2_relu=tf.nn.relu(conv2,name="relu_conv2")
    #conv3_w3=create_weights(filters[2])
    conv3_w3=tf.get_default_graph().get_tensor_by_name("Variable_2:0")  
    conv3=tf.nn.conv2d(conv2_relu,filter=conv3_w3,strides=strides[2],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',  name="COnv3")
    conv3_relu=tf.nn.relu(conv3,name="relu_conv3")
    #conv4_w4=create_weights(filters[3])
    conv4_w4=tf.get_default_graph().get_tensor_by_name("Variable_3:0") 
    conv4=tf.nn.conv2d(conv3_relu,filter=conv4_w4,strides=strides[3],padding=padding,use_cudnn_on_gpu=True,data_format='NHWC',   name="COnv4")
    conv4_relu=tf.nn.relu(conv4,name="relu_conv4")
    
    conv4_flat=tf.reshape(tensor=conv4_relu, shape=[-1,sh[6]*sh[7]*filters[3][3]], name="conv4_flt")
    print(conv4_flat)
    #wfc1=create_weights((sh[6]*sh[7]*filters[3][3],int(fcl[0])))
    wfc1=tf.get_default_graph().get_tensor_by_name("Variable_4:0")
    fc1=tf.matmul(conv4_flat, wfc1, name="fc1")
    fc1_relu=tf.nn.relu(fc1,"relu_fc1")
    fc1_drop=tf.nn.dropout(fc1_relu, keep_prob=drop_rate, seed=10, name="dropout_fc1")
    #wfc2=create_weights((int(fcl[0]),num_classes))
    wfc2=tf.get_default_graph().get_tensor_by_name("Variable_5:0")
    logits=tf.matmul(fc1_drop, wfc2, name="fc2")
    
    return tf.nn.softmax(logits),new_saver,x
    
    