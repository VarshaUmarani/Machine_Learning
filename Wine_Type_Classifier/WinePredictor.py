# Classifier : K Nearest Neighbour
# DataSet : Wine Predictor Dataset
# Features : Alcohol, Malic acid, Ash,Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins,
# Color intensity, Hue, OD280/OD315 of diluted wines, Proline
# Labels : Class 1, Class 2, Class 3
# Training Dataset : 70% of 178 Entries
# Testing Dataset : 30% of 178 Entries

# ===================
# import statements 
# ===================
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictorClassifier():
	# Load dataset
	wine_dataset = datasets.load_wine()

	# print the names of the features
	print(wine_dataset.feature_names)

	# print the label species (class_0, class_1, class_2)
	print(wine_dataset.target_names)

	# print the wine data (top 5 records)
	print(wine_dataset.data[0:5])

	# print the wine labels (0:Class_0,1:Class_1,2:Class_3)
	print(wine_dataset.target)

	# Split dataset into training set and test set
	# 70% for training and 30% for testing
	X_train,X_test,Y_Train,Y_test = train_test_split(wine_dataset.data,wine_dataset.target,test_size=0.3)

	# Create KNN Classifier
	Knn = KNeighborsClassifier(n_neighbors=3)

	# Train the model using the training sets
	Knn.fit(X_train,Y_Train)

	# Predict the response for test dataset
	Y_pred = Knn.predict(X_test)

	# Model Accuracy, how often is the classifier correct?
	print("Accuracy : ",metrics.accuracy_score(Y_test,Y_pred))

def main():
	print("-------------------- Wine Predictor Application using K Nearest Neighbor Algorithm --------------------")

	WinePredictorClassifier()

if __name__ == "__main__":
	main()