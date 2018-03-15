
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# params
weight_decay = 0.01
training_epochs = 5000
display_step = 50
batch_size = 500
image_dim = 32 * 32
learning_rate = 0.1

def get_data():
	with np.load("notMNIST.npz") as data:
		Data, Target = data ["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]

		trainData = trainData.reshape(trainData.shape[0], image_dim)
		validData = validData.reshape(validData.shape[0], image_dim)
		testData = testData.reshape(testData.shape[0], image_dim)
		return trainData, trainTarget, validData, validTarget, testData, testTarget


trainData, trainTarget, validData, validTarget, testData, testTarget = get_data()
num_samples = trainData.shape[0]

X = tf.placeholder(tf.float32, shape=(None, image_dim))
Y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.ones((image_dim, 1)), name="weight")
b = tf.Variable(tf.ones(1), name="bias")

z = tf.add(tf.matmul(X, W), b)

prediction = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
prediction_10 = tf.cast(tf.round(prediction * 10), tf.float64)

correct = tf.reduce_sum(tf.cast(tf.equal(prediction_10, tf.cast(10 * Y, tf.float64)), tf.float64))
accuracy = tf.cast(correct, tf.float64) / tf.cast(tf.shape(prediction_10)[0], tf.float64)


#lD = tf.reduce_sum(-1 * p * tf.log(q) - (1 - p) * tf.log(1 - q))
lD = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=z))
lW = weight_decay * tf.norm(W) / 2
cost = lD + lW

training_loss_for_plot = list()
validation_loss_for_plot = list()

training_accuracy_for_plot = list()
validation_accuracy_for_plot = list()

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=cost)
init = tf.global_variables_initializer()
print ("learning rate: " + str(learning_rate))

with tf.Session() as sess:
	sess.run(init)
	num_batches = int(trainData.shape[0] / batch_size)
	for epoch in range(training_epochs):
		for i in range(num_batches):
			trainBatchi = trainData[i*batch_size: (i+1) * batch_size]
			trainTargeti = trainTarget[i*batch_size: (i+1) * batch_size]
			sess.run(optimizer, feed_dict={X: trainBatchi, Y: trainTargeti})

'''
plot the training and validation curves for both cross-entropy loss and classification accuracy vs. the number of epochs.
Report the best test classification accuracy obtained from the logistic regression model.
'''
		# loss calculation
		train_loss = sess.run(cost, feed_dict={X: trainData, Y: trainTarget})
		validation_loss = sess.run(cost, feed_dict={X: validData, Y: validTarget})

		training_loss_for_plot.append(train_loss)
		validation_loss_for_plot.append(validation_loss)

		# accuracy calculation
		train_acc = sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget})
		validation_acc = sess.run(accuracy, feed_dict={X: validData, Y: validTarget})

		training_accuracy_for_plot.append(train_acc)
		validation_accuracy_for_plot.append(validation_acc)

	test_acc = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget})
	print ("Test accuracy: " + str(test_acc))


# Plot loss vs number of training steps
steps = np.linspace(0, training_epochs, num=training_epochs)
fig = plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, training_loss_for_plot, "r-")
plt.plot(steps, validation_loss_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])

plt.figure()
axes = plt.gca()
fig.patch.set_facecolor('white')
plt.plot(steps, training_accuracy_for_plot, "r-")
plt.plot(steps, validation_accuracy_for_plot, "c-")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
red_patch = mpatches.Patch(color='red', label='Training Set')
cyan_patch = mpatches.Patch(color='cyan', label='Validation Set')
plt.legend(handles=[red_patch, cyan_patch])



plt.show()

