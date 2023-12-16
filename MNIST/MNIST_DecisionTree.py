# importing the libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def MNIST_Decision_Tree(fileName):

	# Step 1 : load the data from the file
	data = pd.read_csv(fileName)

	# Step 2 : Prepare and manipulate the data
	x = data.iloc[:,1:]  # Labels
	y = data.iloc[:,0]   # Pixels

	# Split the data for training and testing 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

	# Create object of Decision Tree Classifier
	dt = DecisionTreeClassifier()

	# Step 3 : Train the model
	dt.fit(x_train,y_train)

	# Accuracy of training with decision tree
	print("Training accuracy using Decision Tree Classifier : ",dt.score(x_train,y_train) * 100)

	# Accuracy of testing with decision tree
	print("Testing accuracy using Decision Tree Classifier : ",dt.score(x_test,y_test) * 100)

def main():
	print("------------------------- Machine Learning Application of MNIST using Decision Tree Classifier -------------------------")

	MNIST_Decision_Tree("MNIST.csv")

if __name__ == "__main__":
	main()
