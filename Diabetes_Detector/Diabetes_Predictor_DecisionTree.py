# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def Diabetes_Predictor_DecisionTree(fileName):
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

	# Create object of Decision Tree Classifier
	tree = DecisionTreeClassifier(random_state=0)

	# Train the model
	tree.fit(X_train, y_train)

	# Accuracy of training with default decision tree
	print("Accuracy on training set : {:.3f}".format(tree.score(X_train, y_train)))

	# Accuracy of testing with default decision tree
	print("Accuracy on test set : {:.3f}".format(tree.score(X_test, y_test)))

	# Create object of Decision Tree Classifier by changing the parameters
	tree = DecisionTreeClassifier(max_depth=3, random_state=0)

	# Train the model
	tree.fit(X_train, y_train)

	# Accuracy of training with changing the parameters of decision tree
	print("Accuracy on training set : {:.3f}".format(tree.score(X_train, y_train)))

	# Accuracy of testing with changing the parameters of decision tree
	print("Accuracy on test set : {:.3f}".format(tree.score(X_test, y_test)))

	# Feature importances of Diabetes dataset
	print("Feature importances :\n{}".format(tree.feature_importances_))

	def plot_feature_importances_diabetes(model):
		plt.figure(figsize=(8,6))
		n_features = 8
		plt.barh(range(n_features), model.feature_importances_, align='center')
		diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
		plt.yticks(np.arange(n_features), diabetes_features)
		plt.xlabel("Feature importance")
		plt.ylabel("Feature")
		plt.ylim(-1, n_features)
		plt.show()

	plot_feature_importances_diabetes(tree)

def main():
	print("------------------------- Machine Learning Application of Diabetes predictor using Decision Tree -------------------------")

	Diabetes_Predictor_DecisionTree('Diabetes.csv')

if __name__ == "__main__":
	main()