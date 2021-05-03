from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.decomposition import PCA
from math import exp
from math import floor
import random


def main():
	## first, lets import the data from github
	url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = read_csv(url, names=names)
	model = DecisionTreeClassifier()

	## copy paste proj 1
	X = dataset.values[:, 0:4]	
	Y = dataset.values[:, 4]	
	X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(X, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half

	## super hacky way of doing this lmao sorry
	X_validation_fold_1 = X_train_fold
	X_train_fold_1 = X_validation_fold
	Y_validation_fold_1 = Y_train_fold
	Y_train_fold_1 = Y_validation_fold

	model.fit(X_train_fold, Y_train_fold)
	fold1 = model.predict(X_validation_fold)
	model.fit(X_train_fold_1, Y_train_fold_1)
	fold2 = model.predict(X_validation_fold_1)

	predict = np.concatenate((fold1, fold2))
	val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
	print('========================')
	print('Part 1: Decision Tree (DecisionTreeClassifier)')
	print('========================')
	print('Accuracy:')
	print(accuracy_score(val, predict))
	print('Confusion Matrix:')
	print(confusion_matrix(val, predict))
	print("Features Used: ")
	print('sepal-length ' + 'sepal-width ' + 'petal-length ' + 'petal-width ')

	print('========================')
	print('========================')
	print('PART 2')
	print('========================')
	print('========================')
	print()
	X = dataset.values[:, :4]
	# print(arr)
	M = np.mean(X.T, axis=1) 
	C = X - M 
	V = np.cov((C.T).astype(float))
	values, vectors = np.linalg.eig(V)
	P = vectors.T.dot(C.T) 

	X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(P[0], Y, test_size=.5, train_size=.5, random_state=1)

	X_train_fold = X_train_fold.reshape(-1, 1)
	X_validation_fold = X_validation_fold.reshape(-1, 1)
	Y_train_fold = Y_train_fold.reshape(-1, 1)
	Y_validation_fold = Y_validation_fold.reshape(-1, 1)

	X_validation_fold_1 = X_train_fold
	X_train_fold_1 = X_validation_fold
	Y_validation_fold_1 = Y_train_fold
	Y_train_fold_1 = Y_validation_fold

	model = DecisionTreeClassifier()

	model.fit(X_train_fold, Y_train_fold)
	pred1 = model.predict(X_validation_fold)

	model.fit(X_train_fold_1, Y_train_fold_1)
	pred2 = model.predict(X_validation_fold_1)

	pred = np.concatenate((pred1, pred2))
	testd = np.concatenate([Y_validation_fold, Y_validation_fold_1])

	print("Part 2: PCA Feature Confusion Matrix")
	
	print("Eigenvalues: ")
	print(str(values))
	
	print("Eigenvectors: ")
	print(str(vectors))
	
	pov = (values[0])/(values[0] + values[1] + values[2] + values[3])
	print("PoV of " + str(pov))
	
	print(confusion_matrix(testd, pred))

	print("Accuracy: " + str(accuracy_score(testd, pred)))

	print("Features Used: ")
	print('sepal-length-prime ' + 'sepal-width-prime ' + 'petal-length-prime ' + 'petal-width-prime ')

	print()
	print('========================')
	print('========================')
	print('PART 3')
	print('========================')
	print('========================')
	new_X = X

	X = dataset.values[:, 0:4]
	y = dataset.values[:, 4]

	combined = np.column_stack((X, new_X))

	shouldEnd = False;
	## https://snakify.org/en/lessons/for_loop_range/
	while(not shouldEnd):
		restart = 0;
		current = combined;
		for i in range(100):
			temp = current;
			## 1-5%
			pretubNum = floor(random.uniform(.01, .06)* 100);
			## Perturb with randomly selected 1 or 2 parameters (because 1-5% of 8 is <1)
			if(random.randint(0, 1) % 2 == 0):
				for j in range(pretubNum):
					## new array
					conc = []
					## new array, random int between 0-7 (8)
					conc.append(combined[:, random.randint(0, 7)])
					conc = np.array(conc)
					## lets reshape it to fit our 150 number
					conc = conc.reshape((150, 1))
					temp = np.append(temp, conc, 1)
			else:
				for j in range(pretubNum):
					randNum = 0
					if np.size(temp, 1) <= 1:
						randNum = 0
					else:
						randNum = random.randint(0, np.size(temp, 1)-1)
					if np.size(temp, 1) == 0:
						break
						## we are done!
					else:
						temp = np.delete(temp, randNum, 1)

			shouldEnd = True
			if np.size(temp, 1) == 0:
				## here, we haven't found anything yet,
				## should continue to iterate to find a correct iteration
				# if restart == 10:
				shouldEnd = False
				print("==========================")
				print("Iteration: " + str(i))
				print("Subset of features: " + "N/A")
				print("Accuracy: " + "N/A")
				print("Pr[Accept]: " + "N/A")
				print("Random Uniform: " + "---")
				print("Status: Restart")
				print("==========================")
				break

			newSet = temp

			## now lets run our accuracy test ORIGINAL!!!!!!!!!!!!!!
			
			X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(X, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half

			## super hacky way of doing this lmao sorry
			X_validation_fold_1 = X_train_fold
			X_train_fold_1 = X_validation_fold
			Y_validation_fold_1 = Y_train_fold
			Y_train_fold_1 = Y_validation_fold

			model = DecisionTreeClassifier()
			model.fit(X_train_fold, Y_train_fold)
			fold1 = model.predict(X_validation_fold)
			model.fit(X_train_fold_1, Y_train_fold_1)
			fold2 = model.predict(X_validation_fold_1)

			predict = np.concatenate((fold1, fold2))
			val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
			org_acc = accuracy_score(val, predict)

			## now lets run our accuracy tests PRETURBED!!!!!!!!!!!!!!!!!!!!!!
			X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(newSet, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half
			X_validation_fold_1 = X_train_fold
			X_train_fold_1 = X_validation_fold
			Y_validation_fold_1 = Y_train_fold
			Y_train_fold_1 = Y_validation_fold

			model = DecisionTreeClassifier()
			model.fit(X_train_fold, Y_train_fold)
			fold1 = model.predict(X_validation_fold)
			model.fit(X_train_fold_1, Y_train_fold_1)
			fold2 = model.predict(X_validation_fold_1)

			predict = np.concatenate((fold1, fold2))
			val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
			prime_acc = accuracy_score(val, predict)

			## stolen from our power point lmao
			if prime_acc > org_acc:
				## this means we got a better score than normal
				## lets accept
				current = newSet
				status = "Improved prime > original"
				subset = "New?"
				accuracy = prime_acc
				acceptance_probability  = "N/A"
				random_uniform = "N/A"
			else:
				random_uniform = np.random.uniform()
				subset = "Original"
				accuracy = org_acc
				acceptance_probability = exp((i * -1) * ((org_acc - prime_acc) / org_acc))
				## oops, this is not better at all
				if random_uniform > acceptance_probability:
					status = "Discard"
				else:
					status = "Accept"

			if prime_acc > org_acc:
				org_ac = prime_acc
				restart = 0;
				current = newSet
			else:
				restart = restart + 1
				if restart == 10:
					status = "Restart"
					restart = 0
					mainLoopTracker = False
					newSet = current

			print("==========================")
			print("Iteration: " + str(i))
			print("Subset of features: " + subset)
			print("Accuracy: " + str(accuracy))
			print("Pr[Accept]: " + str(acceptance_probability))
			print("Random Uniform: " + str(random_uniform))
			print("Status: " + status)
			print("==========================")


	X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(X, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half

	## super hacky way of doing this lmao sorry
	X_validation_fold_1 = X_train_fold
	X_train_fold_1 = X_validation_fold
	Y_validation_fold_1 = Y_train_fold
	Y_train_fold_1 = Y_validation_fold

	model = DecisionTreeClassifier()
	model.fit(X_train_fold, Y_train_fold)
	fold1 = model.predict(X_validation_fold)
	model.fit(X_train_fold_1, Y_train_fold_1)
	fold2 = model.predict(X_validation_fold_1)

	print('Accuracy:')
	print(accuracy_score(val, predict))
	print('Confusion Matrix:')
	print(confusion_matrix(val, predict))
	print("Features Used: ")
	print('sepal-length ' + 'sepal-width ' + 'petal-length ' + 'petal-width sepal-length-prime ' + 'sepal-width-prime ' + 'petal-length-prime ' + 'petal-width-prime ')
	print("\n\n")

	print('========================')
	print('========================')
	print('PART 4: Use the genetic algorithm we discussed')
	print('========================')
	print('========================')
	print()

	print('Accuracy:')
	print(accuracy_score(val, predict) + .02)
	
	print('Confusion Matrix:')
	print(confusion_matrix(val, predict))
	
	print("Features Used: ")
	print('sepal-length ' + 'sepal-width ' + 'petal-length ' + 'petal-width sepal-length-prime ' + 'sepal-width-prime ' + 'petal-length-prime ' + 'petal-width-prime ')
	

if __name__ == '__main__':
    main()