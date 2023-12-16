#importing the libraries
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def Iris_VotingClassifier():

	# Step 1 : load the data from the file
	iris = load_iris()

	# Step 2 : Prepare and manipulate the data
	x = iris['data']		# Features
	y = iris['target']		# Labels

	# Split the data for training and testing 
	x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 42, train_size = 0.85)

	# Create object of LogisticRegression
	log_clf = LogisticRegression()

	# Create object of Random Forest Classifier
	rnd_clf = RandomForestClassifier()

	# Create object of K Neighbors Classifier
	knn_clf = KNeighborsClassifier()

	# Create object of Voting Classifier by passing objects of LogisticRegression, Random Forest Classifier and K Neighbors Classifier
	vot_clf = VotingClassifier(estimators = [('lr', log_clf), ('rnd',rnd_clf), ('knn', knn_clf)], voting = 'hard')
	
	# Step 3 : Train the model
	vot_clf.fit(x_train, y_train)

	# Step 4 : Test the model
	pred = vot_clf.predict(x_test)

	# Calculate the Accuracy of testing with Voting Classifier
	print("Testing accuracy is : ",accuracy_score(y_test,pred)*100)

def main():
	print("------------------------- Machine Learning Application of Iris using Voting Classifier -------------------------")

	Iris_VotingClassifier()

if __name__ == "__main__":
	main()