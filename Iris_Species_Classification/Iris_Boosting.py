# importing the libraries
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score

def Iris_Boosting():
	# Load data
	iris =  datasets.load_iris()

	X = iris.data
	Y = iris.target

	# Split dataset into training set and test set
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) # 70% training and 30% test

	# Create adaboost classifer object
	abc = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,learning_rate=1)

	# Train Adaboost Classifer
	model = abc.fit(X_train, Y_train)

	# Predict the response for test dataset
	Y_pred = model.predict(X_test)

		# Accuracy of training with AdaBoost Classifier
	print("Training accuracy using bagging classifier : ",abc.score(X_train,Y_train)*100)

	# Accuracy of testing with AdaBoost Classifier
	print("Testing accuracy using bagging classifier : ",accuracy_score(Y_test,Y_pred)*100)


def main():
	print("------------------------- Machine Learning Application of Iris using AdaBoost Classifier -------------------------")

	Iris_Boosting()

if __name__ == "__main__":
	main()