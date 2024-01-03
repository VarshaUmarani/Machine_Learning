# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  25-October-2023
# About:  Implementing K Neighbors Classifier model to predict Iris Species.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Consider below characteristics of Machine Learning Application :
# Classifier : K Neighbors Classifier
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa , Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 105 Entries
# Testing Dataset : 45 Entries

# Required Python Packages
import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class KNN:
	@classmethod
	def CalculateDistance(cls,X,Y):
		return distance.euclidean(X,Y)

	def fit(self,train_data,train_target):
		self.train_data = train_data
		self.train_target = train_target

	def predict(self,test_data):
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
INPUT_PATH = "Iris.csv"

# Headers
HEADERS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

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

def encode_data(dataset):
	"""
	encode the data of the dataset
	:param dataset: Pandas dataframe
	:return: updated data
	"""
	sepal_length = dataset.sepal_length
	sepal_width = dataset.sepal_width
	petal_length = dataset.petal_length
	petal_width = dataset.petal_width
	species = dataset.species

	features = list(zip(sepal_length,sepal_width,petal_length,petal_width))

	labelobj = preprocessing.LabelEncoder()
	targets = labelobj.fit_transform(species)

	return features, targets

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

def K_neighbors_classifier(features, target):
	"""
	To train the K Neighbors classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of K Neighbors classifier
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

	features,target = encode_data(dataset)

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(features, target, 0.7)

	# Create instance of K Neighbors Classifier
	trained_model = K_neighbors_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Iris Prediction using User - defined K Neighbors Classifier -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #