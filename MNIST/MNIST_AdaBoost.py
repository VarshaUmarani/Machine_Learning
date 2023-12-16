# importing the libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def MNIST_AdaBoostingClassifier(x_train, x_test, y_train, y_test):
	# First way : Create object of Decision Tree Classifier
	obj = DecisionTreeClassifier(max_depth=10,random_state=4)
	adb = AdaBoostClassifier(obj,n_estimators =100, learning_rate=1)

	# Second way : Create object of AdaBoost Classifier
	#adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 100, learning_rate = 1)

	# Step 3 : Train the model
	adb.fit(x_train,y_train)

	# Accuracy of training with AdaBoost Classifier
	print("Training accuracy using bagging classifier : ",adb.score(x_train,y_train)*100)

	# Accuracy of testing with AdaBoost Classifier
	print("Testing accuracy using bagging classifier : ",adb.score(x_test,y_test)*100)

def MNIST(fileName):

	# Step 1 : load the data from the file
	data = pd.read_csv(fileName)

	# Step 2 : Prepare and manipulate the data
	x = data.iloc[:,1:]  # Labels
	y = data.iloc[:,0]   # Pixels

	# Split the data for training and testing 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

	MNIST_AdaBoostingClassifier(x_train, x_test, y_train, y_test)

def main():
	print("------------------------- Machine Learning Application of MNIST using AdaBoost Classifier -------------------------")

	MNIST("MNIST.csv")

if __name__ == "__main__":
	main()
