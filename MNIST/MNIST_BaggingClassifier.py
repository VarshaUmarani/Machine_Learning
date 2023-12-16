# importing the libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

def MNIST_DecisionTree(x_train, x_test, y_train, y_test):
	# Create object of Decision Tree Classifier
	dt = DecisionTreeClassifier()

	# Step 3 : Train the model
	dt.fit(x_train,y_train)

	# Accuracy of training with decision tree
	print("Training accuracy using Decision Tree Classifier : ",dt.score(x_train,y_train) * 100)

	# Accuracy of testing with decision tree
	print("Testing accuracy using Decision Tree Classifier : ",dt.score(x_test,y_test) * 100)

def MNIST_Random_Forest(x_train, x_test, y_train, y_test):
	# Random Forest - Ensemble of Descision Trees
	rf = RandomForestClassifier(n_estimators=20)

	# Train the model using object of Random Forest Classifier
	rf.fit(x_train,y_train)

	# Accuracy of training with Random Forest Classifier
	print("Training accuracy using Random Forest Classifier : ",rf.score(x_train,y_train)*100)

	# Accuracy of testing with Random Forest Classifier
	print("Testing accuracy using Random Forest Classifier : ",rf.score(x_test,y_test)*100)

def MNIST_Bagging_Classifier(x_train, x_test, y_train, y_test):
	# Bagging Classifier - Ensemble Learning Technique
	bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
	
	# Train the model using object of Bagging Classifier
	bg.fit(x_train,y_train)
	
	# Accuracy of training with Bagging Classifier
	print("Training accuracy using Bagging Classifier : ",bg.score(x_train,y_train)*100)

	# Accuracy of testing with Bagging Classifier
	print("Testing accuracy using Bagging Classifier : ",bg.score(x_test,y_test)*100)

def MNIST(fileName):

	# Step 1 : load the data from the file
	data = pd.read_csv(fileName)

	# Step 2 : Prepare and manipulate the data
	x = data.iloc[:,1:]  # Labels
	y = data.iloc[:,0]   # Pixels

	# Split the data for training and testing 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

	MNIST_DecisionTree(x_train, x_test, y_train, y_test)

	MNIST_Random_Forest(x_train, x_test, y_train, y_test)

	MNIST_Bagging_Classifier(x_train, x_test, y_train, y_test)

def main():
	print("------------------------- Machine Learning Application of MNIST using Decision Tree Random Forest and Bagging Classifier -------------------------")

	MNIST("MNIST.csv")

if __name__ == "__main__":
	main()
