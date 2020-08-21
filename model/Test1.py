import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [9, 10]])
C = tf.matmul(A, B)

# tf.Tensor([[23 26][51 58]], shape=(2, 2), dtype=int32)
print(C)


