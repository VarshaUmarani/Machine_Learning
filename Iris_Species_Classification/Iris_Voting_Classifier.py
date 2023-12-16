# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  25-November-2023
# About:  Implementing Voting Classifier model to predict Iris Species.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Consider below characteristics of Machine Learning Application :
# Classifier : Voting Classifier
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa, Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 105 Entries
# Testing Dataset : 45 Entries
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

def voting_classifier(features, target):
	"""
	To train the Voting Classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of Voting Classifier
	"""
	# Create object of LogisticRegression
	log_clf = LogisticRegression()

	# Create object of Random Forest Classifier
	rnd_clf = RandomForestClassifier()

	# Create object of K Neighbors Classifier
	knn_clf = KNeighborsClassifier()

	# Create object of Voting Classifier by passing objects of LogisticRegression, Random Forest Classifier and K Neighbors Classifier
	clf = VotingClassifier(estimators = [('lr', log_clf), ('rnd',rnd_clf), ('knn', knn_clf)], voting = 'hard')
	
	# Train the model
	clf.fit(features,target)

	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	dataset = read_data(INPUT_PATH)

	# Print volume of dataset
	print("Volume of dataset is : ",len(dataset))

	dataset_statistics(dataset)

	features,target = encode_data(dataset)

	# Split the dataset into training and testing
	train_x, test_x, train_y, test_y = split_dataset(features, target, 0.7)

	# Create instance of Voting Classifier
	trained_model = voting_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)
	predictions = trained_model.predict(test_x)
	
	for i in range(0, 5): 
		print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
	
	print("Training Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))*100)
	print("Testing Accuracy :: ", accuracy_score(test_y, predictions)*100)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
	print("------------------------- Machine Learning Application of Iris Prediction using Voting Classifier -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #