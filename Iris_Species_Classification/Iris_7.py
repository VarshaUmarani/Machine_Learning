# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree and KNN algorithm
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa, Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 75 Entries
# Testing Dataset : 75 Entries

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

def KNNAlgorithm():
	Line = "-"*170

	iris = load_iris()

	data = iris.data
	target = iris.target

	print(Line)
	print("Actual Dataset is : ")
	print(Line)

	for i in range(len(iris.target)):
		print("ID : %d\t Features : %s\t Label : %s" %(i,iris.data[i],iris.target[i]))

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

	print(Line)
	print("Training Dataset is : ")
	print(Line)

	for i in range(len(data_train)):
		print("ID : %d\t  Features : %s\t Label : %s" %(i,data_train[i],target_train[i]))

	print(Line)
	print("Testing Dataset is : ")
	print(Line)

	for i in range(len(data_test)):
		print("ID : %d\t  Features : %s\t Label : %s" %(i,data_test[i],target_test[i]))

	print(Line)

	obj = KNN()

	obj.fit(data_train,target_train)

	output = obj.predict(data_test)

	print("Result of Machine Learning Model : ")
	print(Line)

	for i in range(len(data_test)):
		print("ID : %d\t  Expected Result : %s\t Predicted Result : %s" %(i,target_test[i],output[i]))

	print(Line)

	iCnt = 0
	for i in range(len(data_test)):
		if target_test[i] != output[i]:
			iCnt += 1

	print("Number of Wrong predictions by the ML Model : ",iCnt)
	print(Line)

	Accuracy = CalculateAccuracy(target_test,output)
	return Accuracy

def main():
	print("-------------------------User defined KNN Implementation for Iris dataset-------------------------")
	
	ret = KNNAlgorithm()

	print("Accuracy of KNN algorithm is : ",ret,"%")

	print("-"*170)

if __name__ == "__main__":
	main()