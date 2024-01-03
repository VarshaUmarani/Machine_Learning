import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

class KNN:
	@classmethod
	def CalculateDistance(cls,X,Y):
		return distance.euclidean(X,Y)

	def fit(self,data_train,data_test):
		self.data_train = data_train
		self.data_test = data_test

	def predict(self,data_target):
		Predictions = []

		for row in data_target:
			target = self.ShortestDistance(row)
			Predictions.append(target)

		return Predictions

	def ShortestDistance(self,row):
		MinIndex = 0
		MinDistance = KNN.CalculateDistance(row,self.data_train[0])

		for i in range(1,len(self.data_train)):
			Distance = KNN.CalculateDistance(row,self.data_train[i])
			if Distance < MinDistance:
				MinIndex = i
				MinDistance = Distance

		return self.data_test[MinIndex]

def CalculateAccuracy(expected_result,actual_result):
	total = len(expected_result)
	actual = 0

	for i in range(len(expected_result)):
		if actual_result[i] == expected_result[i]:
			actual += 1

	print("Number of Wrong Predictions is : ",(total-actual))
	print("-"*170)

	Accuracy = (actual / total) * 100
	return Accuracy