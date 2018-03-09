
import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.01
training_epochs = 20000
display_step = 50


print ("hello world")
def get_data():
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
		return trainData, trainTarget, validTarget, validTarget, testData, testTarget


trainData, trainTarget, validTarget, validTarget, testData, testTarget = get_data()
trainData = trainData.reshape(trainData.shape[0], 784)
n_samples = trainData.shape[0]
print (trainData.shape)
print (trainTarget.shape)

X = tf.placeholder(tf.float32, shape=(None, 784))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

pred = tf.add(tf.matmul(X, W), b)
print (pred.shape)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

learning_rates = [0.005, 0.001, 0.0001]
lr = 0.005
batch_sizes = [500, 1500, 3500]
bs = 500
weight_decays = [0.0, 0.001, 0.1, 1.0]

losses = list()
for wd in weight_decays:
	optimizer = tf.train.GradientDescentOptimizer(lr).minimize(
		loss=cost,
		
	)
	init = tf.global_variables_initializer()
	print ("batch size: " + str(bs))

	with tf.Session() as sess:
		sess.run(init)
		num_batches = int(trainData.shape[0] / bs)
		for epoch in range(training_epochs):
			
			c = None
			for i in range(num_batches):
				trainBatchi = trainData[i*bs: (i+1) * bs]
				trainTargeti = trainTarget[i*bs: (i+1) * bs]
				sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})
				if epoch % display_step == 0:
					c = sess.run(cost, feed_dict={X: trainBatchi, Y:trainTargeti})

			if epoch % display_step == 0:	
				print("Epoch: " + str(epoch) + ", cost: " + str(c))

		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		print("Train cost: " + str(train_loss))
		losses.append(train_loss)


print (losses)


















































