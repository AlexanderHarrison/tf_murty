import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

c = np.load("profiledata.npy")
murty_op_module = tf.load_op_library("./libmurtyop.so")
murty_op_module.murty([[1,2,3],[4,5,6]], 10) # init module

stop = 1000
start = timer()
for i, mat in enumerate(c):
    murty_op_module.murty(mat, 10)
    if i == stop:
        break
end = timer()
print(str((end - start) * 1000 / (stop)) + " milliseconds per iteration")
