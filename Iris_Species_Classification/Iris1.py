# Load iris dataset from sklearn.datasets

from sklearn.datasets import load_iris

def main():
	dataset = load_iris()

	print("Features of dataset : ")
	print(dataset.feature_names)

	print("Target names of dataset : ")
	print(dataset.target_names)

if __name__ == "__main__":
	main()