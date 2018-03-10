import tensorflow as tf
import numpy as np



# params
weight_decay = 0.01
#learning_rate = 0.005
training_epochs = 5000
display_step = 50
batch_size = 500


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
		return trainData, trainTarget, validData, validTarget, testData, testTarget



trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()

p = tf.placeholder(tf.float32, shape=(None, 784))
logit_q = tf.placeholder(tf.float32, shape=(None, 1))
q = tf.nn.sigmoid(logit_q)
W = tf.Variable(tf.ones((784, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")


prediction = tf.nn.sigmoid(tf.add(tf.matmul(p, W), b))
lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
lD = tf.nn.sigmoid_cross_entropy_with_logits(labels=p, logits=logit_q)

lW = weight_decay * tf.norm(W) / 2

# binary cross entropy function
cost = lD + lW
print (cost)

learning_rates = [0.005, 0.001, 0.0001]
losses = list()

for learning_rate in learning_rates:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).optimize(
		loss=cost,
	)
	quit()
	init = tf.global_variables_initializer()
	print ("batch size: " + str(batch_size))

	with tf.Session() as sess:
		c = None
		for epoch in range(training_epochs):
			trainDataBatchi = trainData[(epoch) * batch_size: (epoch + 1) * batch_size]
			trainTargetBatchi = trainTarget[(epoch) * batch_size: (epoch + 1) * batch_size]
			sess.run(optimizer, feed_dict={
				X:trainDataBatchi,
				Y:trainTargetBatchi
				})
			if epoch % display_step == 0:
					c = sess.run(cost, feed_dict={X: trainBatchi, Y:trainTargeti})

			if epoch % display_step == 0:
				print("Epoch: " + str(epoch) + ", cost: " + str(c))
		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		print("Train cost: " + str(train_loss))
		losses.append(train_loss)


print (losses)






















































