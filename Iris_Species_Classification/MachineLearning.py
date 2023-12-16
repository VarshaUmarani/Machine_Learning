import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

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
		MinIndex = 0
		MinDistance = KNN.CalculateDistance(row,self.train_data[0])

		for i in range(1,len(self.train_data)):
			Distance = KNN.CalculateDistance(row,self.train_data[i])
			if Distance < MinDistance:
				MinDistance = Distance
				MinIndex = i

		return self.train_target[MinIndex]

def CalculateAccuracy(expected_result,actual_result):
	iCnt = 0
	total = len(expected_result)

	for i in range(len(actual_result)):
		if actual_result[i] == expected_result[i]:
			iCnt += 1

	print("Number of Wrong Predictions is : ",total-iCnt)
	print("-"*170)

	Accuracy = (iCnt / total) * 100
	return Accuracy