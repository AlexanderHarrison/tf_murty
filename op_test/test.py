import tensorflow as tf;
import numpy as np;

murty_op_module = tf.load_op_library("./libmurtyop.so")

a = np.array([[1, -2, 3], [4, -5, 6]], dtype="float32")
b = np.array([[1, -2, 3, 4], [4, -5, 6, -1]], dtype="float32")
a = np.array([[1, 2, -3, -5, -2, 1]], dtype="float32")

result_a = murty_op_module.murty(a, 6)
result_b = murty_op_module.murty(b, 20)
result_c = murty_op_module.murty(a, 1)
print(result_a)
print(result_b)
print(result_c)
