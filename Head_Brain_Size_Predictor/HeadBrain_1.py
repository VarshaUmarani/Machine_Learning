# Classifier : Linear Regression
# DataSet : Head Brain Dataset
# Features : Gender, Age, Head size, Brain weight
# Labels : -
# Training Dataset : 237

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User-Defined implementation of Mean
def Mean(Arr):
	Size = len(Arr)
	Sum = 0

	for i in range(len(Arr)):
		Sum += Arr[i]

	return (Sum / Size)

def HeadBrain(Name):
	# Load data
	dataset = pd.read_csv(Name)
	print("Size of dataset is : ",dataset.shape)

	X = dataset["Head Size(cm^3)"].values
	Y = dataset["Brain Weight(grams)"].values

	print("Length of X : ",len(X))
	print("Length of Y : ",len(Y))

	X_Mean = Mean(X)
	Y_Mean = Mean(Y)

	print("Mean of Independent (X) variable is : ",X_Mean)
	print("Mean of Dependent (Y) variable is : ",Y_Mean)

	Numerator = 0
	Denominator = 0

	# m = (Sum(X-X_Mean) * Sum(Y-Y_Mean)) / Sum(X-X_Mean)**2
	for i in range(len(X)):
		Numerator += (X[i] - X_Mean) * (Y[i] - Y_Mean)
		Denominator += (X[i] - X_Mean)**2

	m = Numerator / Denominator
	print("Slope (m) of Regression line is : ",m)

	# y = mx + c
	# c = y - mx
	# c = Y_Mean - m * X_Mean

	c = Y_Mean - (m * X_Mean)
	print("Y-Intercept (c) of Regression line is : ",c)

	# -200 and +200 are just for proper visibility purpose
	X_Start = np.min(X) - 200
	X_End = np.max(X) + 200

	# Display plotting of above points
	x = np.linspace(X_Start,X_End)
	y = m * x + c

	plt.plot(x,y,color='Green',label="Regression Line")
	plt.scatter(X,Y,color='Red',label="Data plot")

	plt.xlabel("Head Size in cm^3")
	plt.ylabel("Brain Weight in gm")

	# legend() function plots the graph and it is stored on RAM
	plt.legend()

	# show() function displays the graph on screen
	plt.show()

	# Findout goodness of fit i.e R Square
	Numerator = 0
	Denominator = 0
	
	# RSquare = ((m * (X[i] + c) - (Y_mean))**2) / ((Y[i] - Y_Mean)**2)
	for i in range(len(X)):
		Y_pred = m * X[i] + c
		Numerator = Numerator + (Y[i] - Y_Mean)**2
		Denominator = Denominator + (Y[i] - Y_pred)**2

	# RSquare = Numerator / Denominator
	Goodness_of_fit = 1 - (Denominator / Numerator)
	print("Goodness of fit using RSquare Method is : ",Goodness_of_fit)

def main():
	print("-------------------------Head Brain dataset using Linear Regression-------------------------")

	name = input("Enter file name of dataset : ")

	HeadBrain(name)

if __name__ == "__main__":
	main()