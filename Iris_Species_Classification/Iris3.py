# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa , Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 147 Entries
# Testing Dataset : 3 Entries

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

def main():
	dataset = load_iris()

	print("Features of dataset : ")
	print(dataset.feature_names)

	print("Target names of dataset : ")
	print(dataset.target_names)

	# Indices of removed elements
	index = [1,51,101]

	# Step 1 & 2
	# dataset.data -> features
	# dataset.target -> targets or labels

	# Training data with removed elements
	train_feature = np.delete(dataset.data,index,axis=0)
	train_target = np.delete(dataset.target,index)

	# Testing data for testing on training data
	test_feature = dataset.data[index]
	test_target = dataset.target[index]

	# Step 3
	# from decision tree classifier
	obj = tree.DecisionTreeClassifier()

	# Step 4
	# Apply traaining data to form tree
	obj.fit(train_feature,train_target)

	# Step 5
	result = obj.predict(test_feature)

	print("Result predicted by ML Application : ",result)
	print("Result Expected : ",test_target)
	
if __name__ == "__main__":
	main()