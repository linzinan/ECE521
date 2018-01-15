import tensorflow as tf
import numpy as np

def euclidean_distance(X, Z):
	x_sq = tf.reshape(tf.reduce_sum(X*X, 1), [-1, 1])
	z_sq = tf.reshape(tf.reduce_sum(Z*Z, 1), [1, -1])
	D = x_sq - 2*tf.matmul(X, tf.transpose(Z)) + z_sq
	return tf.sqrt(D)

sess = tf.Session()


A = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
B = tf.constant([[0, 0], [-1, -1]], dtype=tf.float32)

print (sess.run(euclidean_distance(A,B)))
