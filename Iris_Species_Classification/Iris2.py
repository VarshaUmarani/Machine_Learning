# Display iris dataset loaded from sklearn.datasets

from sklearn.datasets import load_iris

def main():
	dataset = load_iris()

	print("Features of dataset : ")
	print(dataset.feature_names)

	print("Target names of dataset : ")
	print(dataset.target_names)

	print("Iris dataset : ")

	for icnt in range((len(dataset.target))):
		print("ID : %d  	Feature : %s  	Label : %s"%(icnt+1,dataset.data[icnt],dataset.target[icnt]))

if __name__ == "__main__":
	main()