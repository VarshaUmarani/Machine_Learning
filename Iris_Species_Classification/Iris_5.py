# Consider below characteristics of Machine Learning Application :
# Classifier : Decision Tree and KNN algorithm
# DataSet : Iris Dataset
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels : Versicolor, Setosa , Virginica
# Volume of Dataset : 150 Entries
# Training Dataset : 75 Entries
# Testing Dataset : 75 Entries

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def DecisionTree():
	dataset = load_iris()

	data = dataset.data
	target = dataset.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)

	obj = tree.DecisionTreeClassifier()

	obj.fit(data_train,target_train)

	output = obj.predict(data_test)

	Accuracy = accuracy_score(target_test,output)
	return Accuracy

def KNeighbor():
	dataset = load_iris()

	data = dataset.data
	target = dataset.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)

	obj = KNeighborsClassifier()

	obj.fit(data_train,target_train)

	output = obj.predict(data_test)

	Accuracy = accuracy_score(target_test,output)
	return Accuracy

def main():
	Ret = DecisionTree()
	print("Accuracy of Decision Tree Algorithm is : ",Ret*100,"%")

	Ret = KNeighbor()
	print("Accuracy of KNeighborClassifier Algorithm is : ",Ret*100,"%")

if __name__ == "__main__":
	main()