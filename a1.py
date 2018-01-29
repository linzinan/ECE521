import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def euclidean_distance(X, Z):
	x_sq = tf.reshape(tf.reduce_sum(X*X, 1), [-1, 1])
	z_sq = tf.reshape(tf.reduce_sum(Z*Z, 1), [1, -1])
	D = x_sq - 2*tf.matmul(X, tf.transpose(Z)) + z_sq
	return tf.Session().run(tf.sqrt(D))

def nearest_neighbors(distance_matrix, x_star, index, k):
	# x = training data = list
	# k = number of neighbors

	inv_e_dist = 1 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	resp = np.zeros(np.shape(inv_e_dist))
	run_ind = tf.Session().run(indices)
	resp.put(tf.Session().run(indices), 1 / k)
	return resp


def mean_squared_error(prediction, target):
	return ((prediction - target) ** 2).mean() / 2


def k_nearest_neighbors(x_train, y_train, x_test, k):
	predictions = list()

	for x in x_test:
		prediction = knn_predict(x_train, y_train, x, k)
		prediction.append(prediction)

	return predictions









def knn_predict(x_train, y_train, x_test, k):

	inverse_distance_matrix = 1 / distance_matrix
	values, indices = tf.nn.top_k(inverse_distance_matrix, k)



	resp = np.zeros(np.shape(values)[0])
	resp.put(sess.run(indices[x]), 1 / k)


	one_hot = np.zeros(np.shape(values)[0])
	one_hot.put(sess.run(indices[x]), 1 / k)
	return resp, one_hot

def main():
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

	###########
	plt.scatter(Data, Target, c="g", alpha=0.5)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend(loc=2)
	#plt.show()
	###########

	# distance_matrix = (sess.run(euclidean_distance(Data, Target)))
	# distance_matrix_train = (sess.run(euclidean_distance(trainData, trainTarget)))

	# responsibility, one_hot = nearest_neighbors(1, 10, distance_matrix)
	# y_hat = np.transpose(Target) * responsibility
	# target_one_hot = np.transpose(Target) * one_hot
	# mean_squared_error = ((np.transpose(y_hat) - (target_one_hot)) ** 2).mean()

	# print (mean_squared_error)
	#print (Data)
	#print (Target)

	'''
	Part 2. 2 Prediction
	For the dataset data1D, compute the above k-NN prediction function with 
	k = {1, 3, 5, 50}. For each of these values of k, compute and report the 
	training MSE loss, validation MSE loss and test MSE loss. Choose the best
	k using the validation error.
	'''

	k_list = [1, 3, 5, 50]
	# Train MSE
	# train_mse = list()
	# e_dist = euclidean_distance(trainData, trainData)
	# for k in k_list:
	# 	print ("Calculating for k: " + str(k))
	# 	predictions = list()
	# 	for index in range(len(trainData)):
	# 		resp = nearest_neighbors(e_dist, trainData[index], index, k)
	# 		prediction = np.transpose(trainTarget) * resp
	# 		predictions.append(prediction[0][index])

	# 	print (trainData)
	# 	print (predictions)
	# 	mse = mean_squared_error(predictions, trainTarget)
	# 	train_mse.append(mse)

	# print ("train mse: ")
	# print (train_mse)

	# Train MSE
	validate_mse = list()
	e_dist = euclidean_distance(validData, trainData)
	for k in k_list:
		print ("Calculating for k: " + str(k))
		predictions = list()
		for index in range(len(validData)):
			resp = nearest_neighbors(e_dist, validData[index], index, k)
			prediction = np.transpose(trainTarget) * resp
			print (prediction[0])
			print (index)
			predictions.append(prediction[0])
			#print ("for point: " + str(validData[index]) + ", prediction: " + str(prediction))

		print (np.transpose(validData))
		print (predictions)
		mse = mean_squared_error(predictions, validTarget)
		validate_mse.append(mse)

	print ("validation mse: ")
	print (validate_mse)

	# Train MSE
	test_mse = list()
	e_dist = euclidean_distance(testData, trainData)
	for k in k_list:
		print ("Calculating for k: " + str(k))
		predictions = list()
		for index in range(len(testData)):
			resp = nearest_neighbors(e_dist, testData[index], index, k)
			prediction = np.transpose(trainTarget) * resp
			predictions.append(prediction[0][index])
			#print ("for point: " + str(testData[index]) + ", prediction: " + str(prediction))

		print (testData)
		print (predictions)
		mse = mean_squared_error(predictions, testTarget)
		test_mse.append(mse)

	print ("train mse: ")
	print (test_mse)








if __name__ == "__main__":
	main()


























