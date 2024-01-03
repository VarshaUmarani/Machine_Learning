# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  03-December-2023
# About:  Implementing Random Forest Classifier model for Diabetes Predictor
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Consider below characteristics of Machine Learning Application :
# Classifier : Random Forest Classifier
# DataSet : Diabetes Predictor Dataset
# Features : Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Labels : Negative (0) or Positive (1)
# Volume of Dataset : 768 Entries
# Training Dataset : 70 % of 768 Entries
# Testing Dataset : 30 % of 768 Entries
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

def random_forest_classifier(features, target):
	"""
	To train the Random Forest Classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of Random Forest Classifier
	"""
	clf = RandomForestClassifier(n_estimators=100, random_state=0)
	clf.fit(features, target)
	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Feature importances of Diabetes dataset
def plot_feature_importances_diabetes(model, feature_names):
    """
    To plot feature importance graph
    :param model: Trained model with feature_importances_ attribute
    :param feature_names: List of feature names
    :return: None
    """
    plt.figure(figsize=(8, 6))
    n_features = len(feature_names)   # Exclude the target column
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

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

	# Create instance of Random Forest Classifier
	trained_model = random_forest_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

	print("Confusion matrix :: ")
	print(confusion_matrix(test_y, predictions))

	print("Feature importances :\n{}".format(trained_model.feature_importances_))
	feature_names = HEADERS[1:-1]
	plot_feature_importances_diabetes(trained_model, feature_names)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Diabetes Prediction using Random Forest -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #