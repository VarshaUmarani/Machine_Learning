# Classifier : Logistic Regression
# DataSet : Titanic Dataset
# Features : Passenger id,Gender, Age, Fare, Class etc
# Labels : Survived (0) and Non-Survived (1)

# ===================
# import statements 
# ===================
import numpy as np
import pandas as pd
import seaborn as sb
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ====================================
# Machine Learning Operation function
# ====================================
def Titanic_LogisticRegression():
	banner = "-"*80
	# Step - 1 : Load data
	titanic_Data = pd.read_csv('TitanicDataset.csv')

	print("First five records of loaded dataset : ")
	print(titanic_Data.head())

	# It displays Volume of dataset
	print("Total number of records in dataset is : ",len(titanic_Data))
	
	# It displays information of dataset
	# print("Information of dataset : \n",titanic_Data.info())

	# Step - 2 : Analyze the data
	print(banner)
	print("Visualization : Survived and non-survived passengers : ")

	# figure() function used to create a new figure.
	figure()
	countplot(data=titanic_Data,x="Survived").set_title("Survived vs Non-survived")

	# show() looks for all currently active figure objects, and opens interactive windows that display our figures.
	show()

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Sex : ")
	figure()
	countplot(data=titanic_Data,x="Survived",hue="Sex").set_title("Survived vs Non-survived according to Sex")
	show()

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Pclass : ")
	figure()
	countplot(data=titanic_Data,x="Survived",hue="Pclass").set_title("Survived vs Non-survived according to Pclass")
	show()
	
	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Age : ")
	figure()
	titanic_Data["Age"].plot.hist().set_title("Visualization according to Age")
	show()
	print(banner)

	print(banner)
	print("Visualization : Survived vs Non-survived passengers according to Fare : ")
	figure()
	titanic_Data["Fare"].plot.hist().set_title("Survived vs Non-survived according to Fare")
	show()
	print(banner)

	# Step - 3 : Data cleaning
	# It drops the column named as zero in place
	titanic_Data.drop("zero",axis=1,inplace=True)

	print("First 5 entries from loaded dataset after removing 'zero' column : ")
	print(titanic_Data.head())

	Sex = pd.get_dummies(titanic_Data["Sex"])
	print(Sex.head())

	Sex = pd.get_dummies(titanic_Data["Sex"],drop_first=True)
	print("Sex column after modification : ")
	print(Sex.head())

	Pclass = pd.get_dummies(titanic_Data["Pclass"],drop_first=True)
	print("Pclass column after modification : ")
	print(Pclass.head())

	# Concat Sex and Pclass field in our dataset
	titanic_Data = pd.concat([titanic_Data,Sex,Pclass],axis=1)
	print("Data after concatinating Sex and Pclass column : ")
	print(titanic_Data.head())

	# Removing un-necessary fields from dataset
	titanic_Data.drop(["Sex","sibsp","Parch","Pclass","Embarked"],axis=1,inplace=True)
	print("Dataset after removing irrelevent columns : ")
	print(titanic_Data.head())

	# Divide the dataset into x and y
	X = titanic_Data.drop("Survived",axis=1)
	X.columns = X.columns.astype(str)

	Y = titanic_Data["Survived"]

	# Split the data for training and testing
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5)

	obj = LogisticRegression(max_iter=1000)

	# Step - 4 : Train the machine learning model
	obj.fit(X_train,Y_train)

	# Step - 5 : Test the machine learning model
	test_Result = obj.predict(X_test)

	print("Classification report of Logistic Regression is : ")
	print(classification_report(Y_test,test_Result))

	print("Confusion Matrix of Logistic Regression is : ")
	print(confusion_matrix(Y_test,test_Result))
	
	print("Accuracy of Logistic Regression is : ")
	print(accuracy_score(Y_test,test_Result) * 100)

# =====================
# Entry point function
# =====================
def main():
	print("---------- Titanic Survival Prediction Case Study using Logistic Regression ----------")

	Titanic_LogisticRegression()

# ================
# Code starter
# ================
if __name__ == "__main__":
	main()
