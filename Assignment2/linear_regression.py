
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

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
n_samples = trainData.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
learning_rates = [0.005, 0.001, 0.0001]

for lr in learning_rates:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)
		# Fit all training data
		for epoch in range(training_epochs):
			for (x, y) in zip(trainData, trainTarget):
				sess.run(optimizer, feed_dict={X: x, Y: y})

		    # Display logs per epoch step
			if (epoch+1) % display_step == 0:
				c = sess.run(cost, feed_dict={X: trainData, Y:trainTarget})
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
				"W=", sess.run(W), "b=", sess.run(b))
				quit()

		print("Optimization Finished!")
		training_cost = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

		# Graphic display
		plt.plot(trainData, trainTarget, 'ro', label='Original data')
		plt.plot(trainData, sess.run(W) * trainData + sess.run(b), label='Fitted line')
		plt.legend()
		plt.show()





















































