# Consider below characteristics of Machine Learning Application :
# Classifier : K Nearest Neighbour
# DataSet : Play Predictor Dataset
# Features : Weather , Temperature
# Labels : Yes, No
# Training Dataset : 40 Entries
# Testing Dataset : 12 Entries

from KNNModule import *

def PlayPredictor(path):
	# Step 1 -> load data from csv
	data = pd.read_csv(path,index_col=0)
	print("Play Predictor dataset loaded successfully.!")
	print("Volume of dataset is : ",len(data))

	# We can replace these lines by passing these values directly to fit_transform() method
	Weather = data.Weather
	Temperature = data.Temperature
	Play = data.Play

	# Step 2 -> Prepare data
	# Creating object of LabelEncoder 
	labelobj = preprocessing.LabelEncoder()

	# Converting string labels into numbers
	Weather_Encoded = labelobj.fit_transform(Weather)
	Temperature_Encoded = labelobj.fit_transform(Temperature)
	Target = labelobj.fit_transform(Play)

	# Combining Weather and Temperature into single list of tuples
	Features = list(zip(Weather_Encoded,Temperature_Encoded))
	
	# Splitting training and testing data
	data_train,data_test,target_train,target_test = train_test_split(Features,Target,test_size=0.3)

	# Step 3 -> Train model
	obj = KNN()

	# Train the model using training sets
	obj.fit(data_train,target_train)

	# Step 4 -> Test model
	output = obj.predict(data_test)

	print("-"*170)

	for i in range(len(output)):
		print("ID : %d\t Expected Result : %s\t Predicted Result : %s\t" %(i,target_test[i],output[i]))

	print("-"*170)

	# Calculate accuracy of machine learning model
	Accuracy = CalculateAccuracy(target_test,output)
	return Accuracy

def main():
	print("-------------------------User defined KNN implementation for Play Predictor dataset-------------------------")

	file_name = input("Enter the name of file which contains play predictor dataset : ")

	ret = PlayPredictor(file_name)
	print("Accuracy of User Defined KNN Algorithm is : ",ret,"%")

	print("-"*170)

if __name__ == "__main__":
	main()