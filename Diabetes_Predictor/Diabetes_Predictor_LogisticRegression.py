# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def Diabetes_Predictor_Logistic_Regression(fileName):
	# load the data from the file
	diabetes = pd.read_csv(fileName)

	# Display Columns of Dataset
	print("Columns of Dataset : ")
	print(diabetes.columns)

	# Display first five records of Dataset
	print("First 5 records of dataset : ")
	print(diabetes.head())

	print("Dimension of diabetes data : {}".format(diabetes.shape))

	# Split dataset for training and testing
	X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome' ], diabetes[ 'Outcome' ], stratify = diabetes[ 'Outcome' ] ,random_state=66)

	simplefilter(action='ignore', category=FutureWarning)

	# Create object of Logostic Regression and Train the model
	logreg = LogisticRegression(max_iter=2000).fit(X_train, y_train)

	# Accuracy of training with default parameters of Logistic Regression
	print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))

	# Accuracy of testing with default parameters of Logistic Regression
	print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))

	# Create object of Logostic Regression by changing hyper parameter and Train the model
	logreg001 = LogisticRegression(max_iter=2000,C=0.01).fit(X_train, y_train)

	# Accuracy of training with changing hyper parameters of Logistic Regression
	print("Training set accuracy: {:.3f}".format(logreg001.score(X_train, y_train)))
	
	# Accuracy of testing with changing hyper parameters of Logistic Regression
	print("Test set accuracy: {:.3f}".format(logreg001.score(X_test, y_test)))

def main():
	print("------------------------- Machine Learning Application of Diabetes predictor using Logistic Regression -------------------------")

	Diabetes_Predictor_Logistic_Regression('Diabetes.csv')

if __name__ == "__main__":
	main()