import tensorflow as tf;
import numpy as np;

murty_op_module = tf.load_op_library("./libmurtyop.so")

a = np.array([[1.,2.,3.],[4.,5.,6.]], dtype="float32")

print(murty_op_module.murty(a, 10)) 
