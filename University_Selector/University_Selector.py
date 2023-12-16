# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  04-December-2023
# About:  Implementing Logistic Regression using Gradient Descent
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class LogisticRegressionUsingGD:
	@staticmethod
	def sigmoid(x):
		# Activation function used to map any real value between 0 and 1
		return 1 / (1 + np.exp(-x))
	
	@staticmethod
	def net_input(theta, x):
		# Computes the weighted sum of inputs Similar to Linear Regression
		
		return np.dot(x, theta)
		
	def probability(self, theta, x):
		# Calculates the probability that an instance belongs to a particular class
		
		return self.sigmoid(self.net_input(theta, x))

	def cost_function(self, theta, x, y):
		# Computes the cost function for all the training samples
		m = x.shape[0]
		total_cost = -(1 / m) * np.sum(
			y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
				1 - self.probability(theta, x)))
		return total_cost

	def gradient(self, theta, x, y):
		# Computes the gradient of the cost function at the point theta
		m = x.shape[0]
		return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

	def fit(self, x, y, theta):
		"""trains the model from the training data
		Uses the fmin_tnc function that is used to find the minimum for any function
		It takes arguments as
		1) func : function to minimize
		2) x0 : initial values for the parameters
		3) fprime: gradient for the function defined by 'func'
		4) args: arguments passed to the function Parameters
		----------
		x: array-like, shape = [n_samples, n_features]
		Training samples
		y: array-like, shape = [n_samples, n_target_values]
		Target classes
		theta: initial weights
		Returns
		-------
		self: An instance of self"""

		opt_weights = fmin_tnc(func=self.cost_function, x0=theta,fprime=self.gradient,args=(x, y.flatten()))
		self.w_ = opt_weights[0]
		return self

	def predict(self, x):
		""" Predicts the class labels Parameters
		----------
		x: array-like, shape = [n_samples, n_features]
		Test samples
		Returns
		-------
		predicted class labels """

		theta = self.w_[:, np.newaxis]
		return self.probability(theta, x)

	def accuracy(self, x, actual_classes, probab_threshold=0.5):
		"""Computes the accuracy of the classifier
		Parameters
		----------
		x: array-like, shape = [n_samples, n_features]
		Training samples
		actual_classes : class labels from the training data set
		probab_threshold: threshold/cutoff to categorize the samples into different
		classes
		Returns
		-------
		accuracy: accuracy of the model
		"""

		predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
		predicted_classes = predicted_classes.flatten()
		accuracy = np.mean(predicted_classes == actual_classes)
		return accuracy * 100

def load_data(path, header):
	marks_df = pd.read_csv(path, header=header)
	return marks_df

def main():
	print("-------------------------- Machine Learning Application on University Selector using Logistic Regression -------------------------")	

	# load the data from the file
	data = load_data("marks.txt", None)

	# X = feature values, all the columns except the last column
	X = data.iloc[:, :-1]
	
	# y = target values, last column of the data frame
	y = data.iloc[:, -1]

	# filter out the applicants that got admitted
	admitted = data.loc[y == 1]

	# filter out the applicants that din't get admission
	not_admitted = data.loc[y == 0]
	
	# plots
	plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
	plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,label='Not Admitted')
	plt.legend()
	plt.show()

	# preparing the data for building the model
	X = np.array(np.c_[np.ones((X.shape[0], 1)), X])
	Y = np.array(y)
	theta = np.zeros((X.shape[1], 1))

	# Logistic Regression from scratch using Gradient Descent
	model = LogisticRegressionUsingGD()
	model.fit(X, Y, theta)
	accuracy = model.accuracy(X, Y.flatten())
	parameters = model.w_
	print("The accuracy of the model is : {}".format(accuracy))
	print("The model parameters using Gradient descent : ")
	print(parameters)

	# plotting the decision boundary
	# As there are two features
	# wo + w1x1 + w2x2 = 0
	# x2 = - (wo + w1x1)/(w2)
	x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
	y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]	

	plt.plot(x_values, y_values, label='Decision Boundary')
	plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
	plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,label='Not Admitted')
	plt.xlabel('Marks in 1st Exam')
	plt.ylabel('Marks in 2nd Exam')
	plt.legend()
	plt.show()

	# Using scikit-learn
	model = LogisticRegression()
	model.fit(X, Y)
	parameters = model.coef_
	predicted_classes = model.predict(X)
	accuracy = accuracy_score(Y.flatten(),predicted_classes)
	print('The accuracy score using scikit-learn is : {}'.format(accuracy*100))
	print("The model parameters using scikit learn : ")
	print(parameters)

if __name__ == "__main__":
	main();	
