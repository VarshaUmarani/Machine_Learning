# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree and KNN algorithm
# DataSet : Iris Dataset
# Features : Sepal Length, Sepal Width, Petal Length, Petal Width
# Labels : Versicolor, Setosa , Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 75 Entries
# Testing Dataset : 75 Entries

from MachineLearning import *

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