# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  28-October-2023
# About:  Implementing Logistic Regression model for Titanic Survival Predictor.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Consider below characteristics of Machine Learning Application :
# Classifier : Logistic Regression
# DataSet : Titanic Dataset
# Features : Passenger id,Gender, Age, Fare, Class etc
# Labels : Survived (0) and Non-Survived (1)
# Training Dataset : 1309 Entries
# Testing Dataset : 392 Entries

# Required Python Packages
import numpy as np
import pandas as pd
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "Titanic.csv"

# Headers
HEADERS = ["Passengerid", "Age", "Fare", "Sex", "sibsp", "Parch", "zero", "Pclass", "Embarked", "Survived"]

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

def visualize_data(titanic_Data):
	"""
	Visualize the data using matplotlib
	:param dataset:
	:return: None
	"""

	banner = "-"*80
	
	print(banner)
	print("Visualization : Survived and non-survived passengers : ")

	# figure() function used to create a new figure.
	figure()
	countplot(data=titanic_Data,x="Survived").set_title("Survived vs Non-survived")

	# show() looks for all currently active figure objects, and opens interactive windows that display our figures.
	show()

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Sex : ")
	figure()
	countplot(data=titanic_Data,x="Survived",hue="Sex").set_title("Survived vs Non-survived according to Sex")
	show()

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Pclass : ")
	figure()
	countplot(data=titanic_Data,x="Survived",hue="Pclass").set_title("Survived vs Non-survived according to Pclass")
	show()
	
	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Age : ")
	figure()
	titanic_Data["Age"].plot.hist().set_title("Visualization according to Age")
	show()
	print(banner)

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Fare : ")
	figure()
	titanic_Data["Fare"].plot.hist().set_title("Survived vs Non-survived according to Fare")
	show()
	print(banner)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def clean_data(titanic_Data):
	"""
	clean the data
	:param dataset:
	:return: updated dataset
	"""

	# It drops the column named as zero in place
	titanic_Data.drop("zero",axis=1,inplace=True)

	print("First 5 entries from loaded dataset after removing 'zero' column : ")
	print(titanic_Data.head())

	# Removing un-necessary fields from dataset
	titanic_Data.drop(["Sex","sibsp","Parch","Pclass","Embarked"],axis=1,inplace=True)

	return titanic_Data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def prepare_data(titanic_Data):
	"""
	preparing the data before splitting
	:param dataset:
	:return: features,target
	"""

	# Prepare data for splitting
	features = titanic_Data.drop("Survived",axis=1)
	features.columns = features.columns.astype(str)

	target = titanic_Data["Survived"]

	return features, target

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

def logistic_regression(features, target):
	"""
	To train the Logistic Regression with features and target data
	:param features:
	:param target:
	:return: trained instance of Logistic Regression
	"""
	clf = LogisticRegression(max_iter=1000)
	clf.fit(features, target)
	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	titanic_Data = read_data(INPUT_PATH)

	# It displays Volume of dataset
	print("Total number of records in dataset is : ",len(titanic_Data))

	# Get basic statistics of the loaded dataset 
	dataset_statistics(titanic_Data)

	# Analyzing the data by visualization
	visualize_data(titanic_Data)

	# Clean the data
	titanic_Data = clean_data(titanic_Data)

	# Prepare data before splitting
	features, target = prepare_data(titanic_Data)

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(features, target, 0.7)

	# Create instance of Logistic Regression
	trained_model = logistic_regression(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Classification report of Logistic Regression is : ")
	print(classification_report(test_y,predictions))

	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

	print("Confusion matrix :: ")
	print(confusion_matrix(test_y, predictions))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Titanic Survival Prediction using Logistic Regression -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #