# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  26-November-2023
# About:  Implementing AdaBoost Classifier model for MNIST
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "MNIST.csv"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def read_data(path):
	"""
	Read the data into pandas dataframe
	:param path:
	:return:
	"""
	data = pd.read_csv(path)
	return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_headers(dataset):
	"""
	dataset headers
	:param dataset:
	:return:
	""" 
	return dataset.columns.values

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def dataset_statistics(dataset):
	"""
	Basic statistics of the dataset
	:param dataset: Pandas dataframe
	:return: None, print the basic statistics of the dataset
	"""
	print(dataset.describe())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def prepare_dataset(dataset):
	"""
	prepar the dataset for splitting
	:param dataset:
	:return: features, target
	"""
	# Prepare the dataset for splitting
	features = dataset.iloc[:,1:]  # Labels
	target = dataset.iloc[:,0]   # Pixels

	return features,target

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def split_dataset(feature_headers, target_header, train_percentage):
	"""
	Split the dataset with train_percentage
	:param dataset:
	:param train_percentage:
	:param feature_headers:
	:param target_header:
	:return: train_x, test_x, train_y, test_y
	"""
	# Split dataset into train and test dataset
	train_x, test_x, train_y, test_y = train_test_split(feature_headers, target_header, train_size=train_percentage)
	
	return train_x, test_x, train_y, test_y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ada_boosting_classifier(features, target):
	"""
	To train the AdaBoost Classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of AdaBoost Classifier
	"""
	# First way : Create object of Decision Tree Classifier
	obj = DecisionTreeClassifier(max_depth=10,random_state=4)
	clf = AdaBoostClassifier(obj,n_estimators =100, learning_rate=1)

	# Second way : Create object of AdaBoost Classifier
	#adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 100, learning_rate = 1)
	
	clf.fit(features,target)

	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	dataset = read_data(INPUT_PATH)

	# Print volume of dataset
	print("Volume of dataset is : ",len(dataset))

	dataset_statistics(dataset)

	features,target = prepare_dataset(dataset)

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(features, target, 0.7)

	# Create instance of AdaBoost Classifier
	trained_model = ada_boosting_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of MNIST using AdaBoost Classifier -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #