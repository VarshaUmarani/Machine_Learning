# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Author: Varsha Sidaray Umarani
# Date :  04-November-2023
# About:  Implementing Support Vector Machine model to predict the breast cancer.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required Python Packages
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# File Paths
INPUT_PATH = "breast-cancer-wisconsin.csv"
OUTPUT_PATH = "Output.csv"

# Headers
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
"SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

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
	:return: dataset.columns.values
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

def add_headers(dataset, headers):
	"""
	Add the headers to the dataset
	:param dataset:
	:param headers:
	:return: dataset
	"""
	dataset.columns = headers
	return dataset

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def data_file_to_csv():
	"""
	:return: None, write the data to the file
	"""
	# Headers
	headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion", 
	 "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

	# Load the dataset into Pandas data frame
	dataset = read_data(INPUT_PATH)

	# Add the headers to the loaded dataset
	dataset = add_headers(dataset, headers)

	# Save the loaded dataset into csv format
	dataset.to_csv(OUTPUT_PATH, index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def handel_missing_values(dataset, missing_values_header, missing_label):
	"""
	Filter missing values from the dataset
	:param dataset:
	:param missing_values_header:
	:param missing_label:
	:return: updated dataset
	"""
	return dataset [dataset[missing_values_header] != missing_label]

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

def support_vector_machine(features, target):
	"""
	To train the SVM classifier with features and target data
	:param features:
	:param target:
	:return: trained instance of SVM classifier
	"""
	#Create a svm Classifier 
	clf = svm.SVC(kernel='linear') # Linear Kernel

	#Train the model using the training sets 
	clf.fit(features, target)

	return clf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
	# Load the csv file into pandas dataframe
	dataset = read_data(INPUT_PATH)

	# Get basic statistics of the loaded dataset 
	dataset_statistics(dataset)

	# Filter missing values
	dataset = handel_missing_values(dataset, HEADERS[6], '?')
	train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

	# Train and Test dataset size details
	print("Train_x Shape :: ",train_x.shape)
	print("Train_y Shape:: ",train_y.shape)
	print("Test_x Shape :: ",test_x.shape)
	print("Test_y Shape :: ",test_y.shape)

	# Create instance of Support Vector Machine
	trained_model = support_vector_machine(train_x, train_y)
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
	print("------------------------- Machine Learning Application of Breast Cancer Prediction using Support Vector Machine -------------------------")
	main()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
