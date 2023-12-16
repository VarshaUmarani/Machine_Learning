# Consider below characteristics of Machine Learning Application :
# Classifier : K Nearest Neighbour
# DataSet : Play Predictor Dataset
# Features : Whether , Temperature
# Labels : Yes, No
# Training Dataset : 30 Entries
# Testing Dataset : 1 Entry

import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(path,weather_test,temperature_test):
	# Step 1 -> load data
	# Dataframe
	data = pd.read_csv(path)
	print("Play Predictor Dataset loaded Successfully.!!")
	print("Volume of Dataset is : ",len(data))

	# Step 2 -> Prepare data
	feature = ["Weather", "Temperature"]
	print("Feature names are : ",feature)

	Weather = data.Weather
	Temperature = data.Temperature
	Play = data.Play

	# Creating object of LabelEncoder 
	labelobj = preprocessing.LabelEncoder()

	# Converting string labels into numbers
	# Series
	WeatherX = labelobj.fit_transform(Weather)
	TemperatureX = labelobj.fit_transform(Temperature)
	Label = labelobj.fit_transform(Play)

	# Used to print the series
	# print(WeatherX)
	# print(TemperatureX)

	# Combining Weather and Temperature into single list of tuples
	Features = list(zip(WeatherX,TemperatureX))

	# Step 3 -> Train data
	obj = KNeighborsClassifier(n_neighbors=3)

	# Train the model using training sets
	obj.fit(Features,Label)

	# Step 4 -> Test data
	output = obj.predict([[weather_test,temperature_test]])

	if output == 1:
		print("You can play match today.!")
	else:
		print("You cannot play match today.!")

def main():
	print("--------------------------Play Predictor Case Study-------------------------")

	name = input("Enter the name of file which contains dataset : ")

	weather = input("Enter Weather : ")
	temperature = input("Enter Temperature : ")

	if weather.lower() == "overcast":
		weather = 0
	elif weather.lower() == "rainy":
		weather = 1
	elif weather.lower() == "sunny":
		weather = 2
	else:
		print("Error : Invalid input for Weather.!")

	if temperature.lower() == "cool":
		temperature = 0
	elif temperature.lower() == "hot":
		temperature = 1
	elif temperature.lower() == "mild":
		temperature = 2
	else:
		print("Error : Invalid input for Temperature.!")

	PlayPredictor(name,weather,temperature)

if __name__ == "__main__":
	main()