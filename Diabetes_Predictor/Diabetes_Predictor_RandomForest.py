# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def Diabetes_Predictor_Random_Forest(fileName):
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

	# Create object of Random Forest Classifier
	rf = RandomForestClassifier(n_estimators=100, random_state=0)

	# Train the model
	rf.fit(X_train, y_train)

	# Accuracy of training with default parameters of Random Forest
	print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))

	# Accuracy of testing with default parameters of Random Forest
	print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

	# Create object of Random Forest Classifier by changing hyper parameters
	rf1 = RandomForestClassifier(max_depth=3, n_estimators=100,random_state=0)

	# Train the model
	rf1.fit(X_train, y_train)

	# Accuracy of training with default parameters of Random Forest
	print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))

	# Accuracy of testing with default parameters of Random Forest
	print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))

	def plot_feature_importances_diabetes(model):
		plt.figure(figsize=(8,6))
		n_features = 8
		plt.barh(range(n_features), model.feature_importances_, align='center')
		diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
		plt.yticks(np.arange(n_features), diabetes_features)
		plt.xlabel("Feature importance")
		plt.ylabel("Feature")
		plt.ylim(-1, n_features)
		plt.show()

	plot_feature_importances_diabetes(rf)

def main():
	print("------------------------- Machine Learning Application of Diabetes predictor using Random Forest -------------------------")

	Diabetes_Predictor_Random_Forest('Diabetes.csv')

if __name__ == "__main__":
	main()