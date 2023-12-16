# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  28-November-2023
# About:  Implementing K Neighbors Classifier model for Wine Predictor.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Consider below characteristics of Machine Learning Application :
# Classifier : K Nearest Neighbour
# DataSet : Wine Predictor Dataset
# Features : Alcohol, Malic acid, Ash,Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins,
# Color intensity, Hue, OD280/OD315 of diluted wines, Proline
# Labels : Class 1, Class 2, Class 3
# Training Dataset : 70% of 178 Entries
# Testing Dataset : 30% of 178 Entries

# Required Python Packages
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class KNN:
	@classmethod
	def CalculateDistance(cls, X, Y):
		return np.linalg.norm(X - Y)

	def fit(self,train_data,train_target):
		self.train_data = np.array(train_data)
		self.train_target = np.array(train_target)

	def predict(self,test_data):
		test_data = np.array(test_data)
		predictions = []

		for row in test_data:
			target = self.ShortestDistance(row)
			predictions.append(target)

		return predictions

	def ShortestDistance(self,row):
		min_index = 0
		min_distance = KNN.CalculateDistance(row, self.train_data[0])

		for i in range(1, len(self.train_data)):
			distance = KNN.CalculateDistance(row, self.train_data[i])
			if distance < min_distance:
				min_distance = distance
				min_index = i

		return self.train_target[min_index]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "WinePredictor.csv"

# Headers
HEADERS = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
 "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline", "Class"]

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

def K_neighbors_classifier(features, target):
	"""
	To train the K Neighbors Classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of K Neighbors Classifier
	"""
	clf = KNN()
	clf.fit(features, target)
	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	dataset = read_data(INPUT_PATH)

	# Print volume of dataset
	print("Volume of dataset is : ",len(dataset))

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

	# Create instance of K Neighbors Classifier
	trained_model = K_neighbors_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print(f"Actual outcome :: {list(test_y)[i]} and Predicted outcome :: {predictions[i]}")
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Wine Predictior using User - Defined K Neighbors Classifier -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #