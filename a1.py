import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def euclidean_distance(X, Z):
	x_sq = tf.reshape(tf.reduce_sum(X*X, 1), [-1, 1])
	z_sq = tf.reshape(tf.reduce_sum(Z*Z, 1), [1, -1])
	D = x_sq - 2*tf.matmul(X, tf.transpose(Z)) + z_sq
	return tf.sqrt(D)

def nearest_neighbors(x, k, distance_matrix):
	inverse_distance_matrix = 1 / distance_matrix
	values, indices = tf.nn.top_k(inverse_distance_matrix, k)
	print (sess.run(values))
	print (sess.run(indices))
	resp = np.zeros(np.shape(values)[0])
	resp.put(sess.run(indices[x]), 1/k)
	print (resp)




sess = tf.Session()


A = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
B = tf.constant([[0, 0], [-1, -1]], dtype=tf.float32)


np.random.seed(521)
Data = np.linspace(1.0, 10.0, num = 100) [:, np.newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

plt.scatter(trainData, trainTarget, c="g", alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc=2)
#plt.show()

distance_matrix = (sess.run(euclidean_distance(trainData, trainTarget)))

nearest_neighbors(1, 5, distance_matrix)


