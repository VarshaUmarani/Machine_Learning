# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree and KNN algorithm
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa, Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 75 Entries
# Testing Dataset : 75 Entries

from MLModule import *

def KNNAlgorithm():
	Line = "_"*170

	iris = load_iris()

	data = iris.data
	target = iris.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

	obj = KNN()

	obj.fit(data_train,target_train)

	output = obj.predict(data_test)

	print("Result of Machine Learning Model : ")
	print(Line)

	for i in range(len(data_test)):
		print("ID : %d\t\tExpected Result : %s\tPredicted Result : %s" %(i,target_test[i],output[i]))

	print(Line)

	Accuracy = CalculateAccuracy(target_test,output)
	return Accuracy

def main():
	print("-------------------------User defined KNN Implementation for Iris dataset-------------------------")
	
	ret = KNNAlgorithm()

	print("Accuracy of KNN algorithm is : ",ret,"%")

if __name__ == "__main__":
	main()