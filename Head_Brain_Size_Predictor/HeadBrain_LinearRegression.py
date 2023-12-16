# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  28-October-2023
# About:  Implementing Linear Regression model for HeadBrain dataset.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Classifier : Linear Regression
# DataSet : Head Brain Dataset
# Features : Gender, Age, Head size, Brain weight
# Labels : -
# Training Dataset : 237
# Testing Dataset : 15
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "HeadBrain.csv"
TESTING_PATH = "HeadBrain-test.csv"

# Headers
HEADERS = ["Gender", "Age Range", "Head Size(cm^3)", "Brain Weight(grams)"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def read_data(path):
	"""
	Read the data into pandas dataframe
	:param path:
	:return: dataframe
	"""
	data = pd.read_csv(path)
	return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_headers(dataset):
	"""
	dataset headers
	:param dataset:
	:return: dataset.column.values
	""" 
	return dataset.columns.values

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def dataset_statistics(dataset):
	"""
	Basic statistics of the dataset
	:param dataset: Pandas dataframe
	:return: None, print the basic statistics of the dataset
	"""
	print("Headers of dataset : ",get_headers(dataset))
	print(dataset.describe())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def prepare_dataset(dataset, testing_dataset):
	"""
	prepar the dataset for training and testing
	:param dataset:
	:param testing_dataset:
	:return: train_x, train_y, test_data, test_target
	"""
	# Prepare the dataset into train and test dataset
	train_x = dataset["Head Size(cm^3)"].values
	train_y = dataset["Brain Weight(grams)"].values

	# Reshape train_x
	train_x = train_x.reshape(-1,1)

	test_data = testing_dataset["Head Size(cm^3)"].values
	test_target = testing_dataset["Brain Weight(grams)"].values

	# Reshape test_data
	test_data = test_data.reshape(-1,1)

	return train_x, train_y, test_data, test_target

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def linear_regression(features, target):
	"""
	To train the Linear Regression with features and target data
	:param features:
	:param target:
	:return: trained instance of Linear Regression
	"""
	clf = LinearRegression()
	clf.fit(features, target)
	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	dataset = read_data(INPUT_PATH)
	testing_dataset = read_data(TESTING_PATH)

	# Print size of dataset
	print("Size of dataset is : ",dataset.shape)

	# Get basic statistics of the loaded dataset
	dataset_statistics(dataset)

	train_x, train_y, test_x, test_y = prepare_dataset(dataset,testing_dataset)

	# Train and Test dataset size details
	print("Train_x Shape :: ",train_x.shape)
	print("Train_y Shape:: ",train_y.shape)
	print("Test_x Shape :: ",test_x.shape)
	print("Test_y Shape :: ",test_y.shape)

	# Create instance of Linear Regression
	trained_model = linear_regression(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5):
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	RSquare_training = trained_model.score(train_x,train_y)
	print("Value of R Square for training : ",RSquare_training)

	RSquare_testing = trained_model.score(test_x,test_y)
	print("Value of R Square for testing : ",RSquare_testing)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of HeadBrain using Linear Regression -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #