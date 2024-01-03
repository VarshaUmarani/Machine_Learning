# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree and KNN algorithm
# DataSet : Iris Dataset
# Features : Sepal Length, Sepal Width, Petal Length, Petal Width
# Labels : Versicolor, Setosa , Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 75 Entries
# Testing Dataset : 75 Entries

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

def KNNAlgorithm(path):
	data = pd.read_csv(path)

	print("Iris dataset loaded successfully..!!")
	print("Volume of Iris dataset is : ",len(data))

	sepal_length = data.sepal_length
	sepal_width = data.sepal_width
	petal_length = data.petal_length
	petal_width = data.petal_width
	species = data.species

	features = list(zip(sepal_length,sepal_width,petal_length,petal_width))

	labelobj = preprocessing.LabelEncoder()
	targets = labelobj.fit_transform(species)

	data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.5)

	obj = KNN()

	obj.fit(data_train,target_train)

	output = obj.predict(data_test)

	print("-"*170)
	for i in range(len(target_test)):
		print("ID : %d\tExpected Result : %s\tPredicted Result : %s" %(i,target_test[i],output[i]))

	print("-"*170)
	Accuracy = CalculateAccuracy(target_test,output)
	return Accuracy

def main():
	print("-------------------------User defined KNN Implementation for Iris dataset-------------------------")

	name = input("Enter the name of file which contains iris dataset : ")
	
	ret = KNNAlgorithm(name)
	print("Accuracy of KNN Algorithm is : ",ret,"%")

	print("-"*170)	

if __name__ == "__main__":
	main()