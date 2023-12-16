# Classifier : Linear Regression
# DataSet : Head Brain Dataset
# Features : Gender, Age, Head size, Brain weight
# Labels : -
# Training Dataset : 237

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def HeadBrain(Name1,Name2):
	dataset = pd.read_csv(Name1)
	testing_dataset = pd.read_csv(Name2)

	print("Size of our dataset is : ",dataset.shape)

	X = dataset["Head Size(cm^3)"].values
	Y = dataset["Brain Weight(grams)"].values

	test_data = testing_dataset["Head Size(cm^3)"].values
	test_target = testing_dataset["Brain Weight(grams)"].values

	X = X.reshape(-1,1)
	test_data = test_data.reshape(-1,1)

	obj = LinearRegression()

	obj.fit(X,Y)

	Output = obj.predict(test_data)

	print("-"*170)

	for i in range(len(Output)):
		print("ID : %d\t Expected Result : %s\t Predicted_Result : %s" %(i,test_target[i],Output[i]))

	print("-"*170)

	RSquare = obj.score(X,Y)

	print("Value of R Square is : ",RSquare)
	print("-"*170)

def main():
	print("-------------------------Head Brain dataset using Linear Regression-------------------------")

	name1 = input("Enter file name of dataset : ")
	name2 = input("Enter file name of testing dataset : ")

	HeadBrain(name1,name2)

if __name__ == "__main__":
	main()