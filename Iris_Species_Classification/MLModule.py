from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNN:
	@classmethod
	def CalculateDistance(cls,X,Y):
		return distance.euclidean(X,Y)

	def fit(self,trainingData,trainingTarget):
		self.trainingData = trainingData
		self.trainingTarget = trainingTarget

	def predict(self,testingData):
		predictions = []

		for row in testingData:
			target = self.ShortestDistance(row)
			predictions.append(target)

		return predictions

	def ShortestDistance(self,row):
		MinIndex = 0
		MinDistance = KNN.CalculateDistance(row,self.trainingData[0])

		for i in range(1,len(self.trainingData)):
			Distance = KNN.CalculateDistance(row,self.trainingData[i])
			if Distance < MinDistance:
				MinDistance = Distance
				MinIndex = i

		return self.trainingTarget[MinIndex]

def CalculateAccuracy(target_test,actual_result):
	total = len(target_test)
	actual = 0

	for i in range(len(target_test)):
		if target_test[i] == actual_result[i]:
			actual += 1

	Accuracy = (actual / total) * 100
	return Accuracy