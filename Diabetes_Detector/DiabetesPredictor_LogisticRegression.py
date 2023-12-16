# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  03-December-2023
# About:  Implementing Logistic Regression model for Diabetes Predictor
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Consider below characteristics of Machine Learning Application :
# Classifier : Logistic Regression
# DataSet : Diabetes Predictor Dataset
# Features : Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Labels : Negative (0) or Positive (1)
# Volume of Dataset : 768 Entries
# Training Dataset : 70 % of 768 Entries
# Testing Dataset : 30 % of 768 Entries
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "Diabetes.csv"

# Headers
HEADERS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age","Outcome"]

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
	print("Headers of dataset : ",get_headers(dataset))
	print(dataset.describe())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def split_dataset(dataset, train_percentage, feature_headers, target_header):
	"""
	Split the dataset with train_percentage
	:param dataset:
	:param train_percentage:
	:param feature_headers:
	:param target_header:
	:return: train_x, test_x, train_y, test_y
	"""
	# Split dataset into train and test dataset
	train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
	
	return train_x, test_x, train_y, test_y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def logistic_regression(features, target):
	"""
	To train the Logistic Regression with features and target data
	:param features:
	:param target:
	:return: trained instance of Logistic Regression
	"""
	clf = LogisticRegression(max_iter=2000,C=0.01)
	clf.fit(features, target)
	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	diabetes = read_data(INPUT_PATH)

	# Print volume of dataset
	print("Volume of dataset is : ",len(diabetes))

	# Display first five records of Dataset
	print("First 5 records of dataset : ")
	print(diabetes.head())

	print("Dimension of diabetes data : {}".format(diabetes.shape))

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(diabetes, 0.7, HEADERS[1:-1], HEADERS[-1])

	# Create instance of Logistic Regression
	trained_model = logistic_regression(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

	print("Confusion matrix :: ")
	print(confusion_matrix(test_y, predictions))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Diabetes Prediction using Logistic Regression -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #