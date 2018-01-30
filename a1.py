import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib
sess = tf.Session()

def euclidean_distance(X, Z):
	global sess
	x_sq = tf.reshape(tf.reduce_sum(X*X, 1), [-1, 1])
	z_sq = tf.reshape(tf.reduce_sum(Z*Z, 1), [1, -1])
	D = x_sq - 2*tf.matmul(X, tf.transpose(Z)) + z_sq
	matrix = sess.run(tf.sqrt(D))
	return matrix

def nearest_neighbors(distance_matrix, x_star, index, k):
	# x = training data = list
	# k = number of neighbors
	global sess
	inv_e_dist = 1 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	resp = np.zeros(np.shape(inv_e_dist))
	run_ind = sess.run(indices)
	np.put(resp, run_ind, 1.0 / k)
	return 

def nearest_neighbors_part3(distance_matrix, x_star, index, k):
	# x = training data = list
	# k = number of neighbors
	global sess
	inv_e_dist = 1 / distance_matrix[index]
	values, indices = tf.nn.top_k(inv_e_dist, k)
	return (sess.run(indices))
	quit()
	y, idx, count = tf.unique_with_counts(values)
	y, idx, count = sess.run(y), sess.run(idx), sess.run(count)
	count = list(count)
	return y[count.index(max(count))]



def mean_squared_error(prediction, target):
	return ((prediction - target) ** 2).mean() / 2

def get_data():
	Data = np.linspace(1.0, 10.0, num = 100) [:, np.newaxis]
	Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
	return Data, Target

def partition_data(Data, Target):
	np.random.seed(521)
	randIdx = np.arange(100)
	np.random.shuffle(randIdx)
	trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
	validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
	testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
	return trainData, trainTarget, validData, validTarget, testData, testTarget

def test_k_values(data_points, trainData, trainTarget, kval = None):
	if kval == None:
		k_list = [1, 3, 5]
		prediction_list = list()
	else:
		k_list = [kval]
	data_mse = list()
	e_dist = euclidean_distance(data_points, trainData)
	for k in k_list:
		print ("Calculating for k: " + str(k))
		full_rank_resp = list()
		for index in range(len(data_points)):
			resp = nearest_neighbors(e_dist, data_points[index], index, k)
			full_rank_resp.append(resp)

		prediction = np.matmul(np.transpose(trainTarget), np.transpose(full_rank_resp))
		print (prediction)
		mse = mean_squared_error(prediction, trainTarget)
		prediction_list.append(prediction)
		data_mse.append(mse)

	print ("mse: ")
	print (data_mse)
	return data_mse, prediction_list

def test_k_values_part3(data_points, trainData, trainTarget, kval = None):
	if kval == None:
		k_list = [1, 3]
	else:
		k_list = [kval]
	prediction_list = list()
	data_mse = list()
	e_dist = euclidean_distance(data_points, trainData)
	print (np.shape(e_dist))
	for k in k_list:
		print ("Calculating for k: " + str(k))
		full_rank_resp = list()
		for index in range(len(data_points)):
			indices = nearest_neighbors_part3(e_dist, data_points[index], index, k)
			values = list()
			for index in indices:
				values.append(trainTarget[index])
			y, idx, count = tf.unique_with_counts(values)
			y, idx, count = sess.run(y), sess.run(idx), list(sess.run(count))
			full_rank_resp.append(y[count.index(max(count))])
		print (full_rank_resp)

	return data_mse, prediction_list


def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task 
	data = np.load(data_path) / 255.0
	data = np.reshape(data, [-1, 32*32])

	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
			data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
			data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
			target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
			target[rnd_idx[trBatch + validBatch + 1:-1], task]
	return trainData, validData, testData, trainTarget, validTarget, testTarget


def main():
	sess = tf.Session()

	Data, Target = get_data()
	trainData, trainTarget, validData, validTarget, testData, testTarget = partition_data(Data, Target)

	###########
	# plt.scatter(trainData, trainTarget, c="g", alpha=0.5)
	# plt.scatter(validData, validTarget, c="r", alpha=0.5)
	# plt.scatter(testData, testTarget, c="b", alpha=0.5)
	# plt.xlabel("X")
	# plt.ylabel("Y")
	# plt.legend(loc=2)
	
	###########

	'''
	Part 2. 2 Prediction
	For the dataset data1D, compute the above k-NN prediction function with 
	k = {1, 3, 5, 50}. For each of these values of k, compute and report the 
	training MSE loss, validation MSE loss and test MSE loss. Choose the best
	k using the validation error.
	'''


	# Train MSE
	#train_mse, _ = test_k_values(trainData, trainData, trainTarget)

	# Validation MSE
	#print ("Validation MSE")
	#valid_mse, _ = test_k_values(validData, trainData, trainTarget)

	# Test MSE
	#print ("Test MSE")
	#test_mse, _ = test_k_values(testData, trainData, trainTarget)

	# random points MSE
	#print ("X MSE")
	#X = np.linspace(0.0, 11.0, num = 1000)[:, np.newaxis]
	#x_mse, prediction_list = test_k_values(X, trainData, trainTarget)
	#print (x_mse)
	#print (prediction_list)

	# for prediction in prediction_list:
	# 	plt.figure()
	# 	plt.scatter(Data, Target, c = "b", alpha = 0.5)
	# 	plt.plot(X, np.transpose(prediction), c = "g")
	# #plt.show()

	'''
	Part 3. 1 Predicting class label
	Modify the prediction function for regression in section 1 and use 
	majority voting over k nearest neighbors to predict the final class. 
	You may find tf.unique with counts helpful for this task. Include 
	the relevant snippet of code for this task.
	'''
	# task 0
	trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0)

	train_mse, prediction_list = test_k_values_part3(trainData, trainData, trainTarget)
	
	for prediction in prediction_list:
		y, idx, count = tf.unique_with_counts(tf.transpose(prediction))
		print (sess.run(y))
		#print (sess.run(idx))
		print (sess.run(count))





if __name__ == "__main__":
	main()


























