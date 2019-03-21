import tensorflow as tf
import array
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os

class layers():
	
	def __init__(self):
		self.input_layer = []
		self.output_layer = []
		self.seed = 128
		self.rng = np.random.RandomState(self.seed)
		self.layers = 0
		self.weights = {}
		self.biases = {}
		self.neurons = []
		self.hidden_layer = []
		self.cost = 0
		self.optimizer = 0
		self.learning_rate = 0.01
		self.sess = tf.Session()
		self.init = 0
		self.pred = []
		# self.x_train = []
		# self.y_train = []
	
	def Activate(self, activation="relu"):
		if activation == "relu":
			self.hidden_layer[self.layers]=tf.nn.relu(self.hidden_layer[self.layers])
		if activation == "softmax":
			self.output_layer=tf.placeholder(tf.float32, [None, self.neurons[self.layers]])
			# print(self.output_layer.shape)
			# print(self.output_layer[9])
			self.cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hidden_layer[self.layers], labels=self.output_layer))

	def dense_to_one_hot(self,labels_dense, num_classes):
	    num_labels = labels_dense.shape[0]
	    index_offset = np.arange(num_labels) * num_classes
	    labels_one_hot = np.zeros((num_labels, num_classes))
	    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	    return labels_one_hot

	def batch_creator(self,batch_size, x_train,y_train , dataset_name):
	    dataset_length = x_train.shape[0]
	    batch_mask = self.rng.choice(dataset_length, batch_size)
	    # print(batch_mask)
	    # print(x_train.shape)
	    batch_x = eval('x_'+dataset_name)[[batch_mask]].reshape(-1, self.neurons[0])
	    # print(batch_x.shape)
	    if dataset_name == 'train':
	    	batch_y = eval('y_'+dataset_name)[[batch_mask]]
	    	# print(batch_y.shape)
	    	batch_y = self.dense_to_one_hot(batch_y,self.neurons[self.layers])
	    	# print(batch_y)
	    return batch_x, batch_y

	def Input(self,width,height,channels=1):

		self.neurons.append(width * height * channels)
		self.input_layer = tf.placeholder(tf.float32, [None , self.neurons[0]])
		self.hidden_layer.append(self.input_layer)
		# print(self.input_layer)

	def add_layer(self, neurons, activation):
		rows = self.neurons[self.layers]
		columns = neurons
		self.layers+=1
		# print(self.layers)
		self.neurons.append(neurons)
		self.weights['hidden'+str(self.layers)] = tf.Variable(tf.random_normal([rows, columns], seed=self.seed))
		self.biases['hidden'+str(self.layers)] = tf.Variable(tf.random_normal([columns], seed=self.seed))
		self.hidden_layer.append(tf.add(tf.matmul(self.hidden_layer[self.layers-1], self.weights['hidden'+str(self.layers)]), self.biases['hidden'+str(self.layers)]))
		self.Activate(activation)

	def compile(self):
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

	def train(self,x_train,y_train,batch_size,epochs):
		os.system("clear")
		for epoch in range(epochs):
			avg_cost = 0
			total_batch = int(x_train.shape[0]/batch_size)
			for i in range(total_batch):
			   batch_x, batch_y = self.batch_creator(batch_size, x_train, y_train, 'train')
			   # print(self.input_layer,self.output_layer)
			   _, c = self.sess.run([self.optimizer,self.cost], feed_dict = {self.input_layer: batch_x, self.output_layer: batch_y})
			   avg_cost += c / total_batch
			print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

	def predict(self, x_test, y_test):
		keep_prob = tf.placeholder(tf.float32) 
		pred_temp = tf.equal(tf.argmax(self.hidden_layer[self.layers], 1), tf.argmax(self.output_layer, 1))
		accuracy = tf.reduce_mean(tf.cast(pred_temp,"float"))
		test_accuracy = self.sess.run(accuracy, feed_dict={self.input_layer:x_test.reshape(-1, self.neurons[0]),self.output_layer:self.dense_to_one_hot(y_test,self.neurons[self.layers]),keep_prob:1.0})
		print("test_accuracy:",test_accuracy)
		predict = tf.argmax(self.hidden_layer[self.layers], 1)
		# print(predict)
		self.pred = predict.eval({self.input_layer: x_test.reshape(-1, self.neurons[0])},session=self.sess)
		fs=f1_score(y_test, self.pred, average = 'weighted')
		print("test_f1-score: ",fs)
		# pred = self.dense_to_one_hot(predict.eval({self.input_layer: x_test.reshape(-1, self.neurons[0])},session=self.sess))
		# y_test=self.dense_to_one_hot(y_test)/
		# print(pred.shape)
		# print(y_test.shape)
		# conf_mat = confusion_matrix(y_test, pred)
		# print(conf_mat)
		# print("Validation Accuracy:", accuracy.eval({self.input_layer: , self.output_layer: }))

	def learning_curve(self,estimator,title, X, y, ylim=None, cv=None,n_jobs=None,train_sizes=np.linspace(.1, 1.0, 5)):
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(
		    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
		                 train_scores_mean + train_scores_std, alpha=0.1,
		                 color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
		                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
		         label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
		         label="Cross-validation score")

		plt.legend(loc="best")
		return plt
	
	def confusion(self,y_test):
		cm=confusion_matrix(y_test,self.pred)
		print("Confusion Matrix=")
		print(cm)
		# np.savetxt('lines_CM.txt',cm,fmt='%.2f')	
		# print(f1_score(y_test, self.pred, average='samples'))
		# print(f1_score(y_test, self.pred, average='macro'))
