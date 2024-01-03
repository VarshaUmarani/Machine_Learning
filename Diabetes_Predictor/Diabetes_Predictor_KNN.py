# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def Diabetes_Predictor_KNN(fileName):
	# load the data from the file
	diabetes = pd.read_csv(fileName)

	# Display Columns of Dataset
	print("Columns of Dataset : ")
	print(diabetes.columns)

	# Display first five records of Dataset
	print("First 5 records of dataset : ")
	print(diabetes.head())

	print("Dimension of diabetes data : {}".format(diabetes.shape))

	# Split dataset for training and testing
	X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome' ], diabetes[ 'Outcome' ], stratify = diabetes[ 'Outcome' ] ,random_state=66)

	training_accuracy = []
	test_accuracy = []

	# try n_neighbors from 1 to 10
	neighbors_settings = range(1, 11)

	for n_neighbors in neighbors_settings:
		# Build the model using K NeighboursClassifier
		knn = KNeighborsClassifier(n_neighbors=n_neighbors)

		# Train the model
		knn.fit(X_train, y_train)

		# Record training set accuracy
		training_accuracy.append(knn.score(X_train, y_train))

		# Record test set accuracy
		test_accuracy.append(knn.score(X_test, y_test))

	plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
	plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("n_neighbors")
	plt.legend()
	plt.savefig('knn_compare_model')
	plt.show()

	# Choose the hyper parameter (n_neighbors) value with high accuracy from above graph
	knn = KNeighborsClassifier(n_neighbors=9)

	# Train the model
	knn.fit(X_train, y_train)

	print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

	print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test,y_test)))

def main():
	print("------------------------- Machine Learning Application of Diabetes predictor using Decision Tree -------------------------")

	Diabetes_Predictor_KNN('Diabetes.csv')

if __name__ == "__main__":
	main()