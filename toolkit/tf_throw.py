import tensorflow as tf
import numpy as np

v= tf.Variable(initial_value=tf.truncated_normal(shape=(5,5,3,32), mean=0, stddev=np.sqrt(2/8), dtype="float32", seed=10, name="conv1_w"),dtype="float32")

init=tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)
    print(type(s.run(v)))
    
    