from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
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
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import numpy as np
from numpy import array
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

discount = 1


print("============================================")
print("================= PART 1 Policy Iteration =====================")
print("============================================")

iteration_policy = 0

## Initalize a policy pi arbitrarily
## using 5x5 instead of 4x4 like we did inside of class
gridworld_policy = np.zeros((5,5))

## using 5x5 instead of 4x4 like we did inside of class
gridworld_policy_prev = np.ones((5,5))


## repeat
while not np.array_equal(gridworld_policy, gridworld_policy_prev):
	if iteration_policy == 0 or iteration_policy == 1 or iteration_policy == 2:
		print("Iteration: " , str(iteration_policy))
		print(gridworld_policy)
	iteration_policy += 1;

	## pi = pi prime
	gridworld_policy_prev = np.copy(gridworld_policy)

	for i in range(0,5):
		for j in range(0,5):
			if i == 0:
				if j != 0 and j != 4:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i+1][j]*0.25 + gridworld_policy_prev[i][j+1]*0.25 + gridworld_policy_prev[i][j-1]*0.25
				elif j == 4:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i+1][j]*0.25 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i][j-1]*0.25
			elif i == 4:
				if j != 0 and j != 4:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i-1][j]*0.25 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i][j+1]*0.25 + gridworld_policy_prev[i][j-1]*0.25
				elif j == 0:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i-1][j]*0.25 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i][j+1]*0.25 + gridworld_policy_prev[i][j]*0.25
			else:
				if j != 0 and j != 4:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i-1][j]*0.25 + gridworld_policy_prev[i+1][j]*0.25 + gridworld_policy_prev[i][j+1]*0.25 + gridworld_policy_prev[i][j-1]*0.25
				elif j == 4:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i-1][j]*0.25 + gridworld_policy_prev[i+1][j]*0.25 + gridworld_policy_prev[i][j]*0.25 + gridworld_policy_prev[i][j-1]*0.25
				elif j == 0:
					gridworld_policy[i][j] =  -1 + gridworld_policy_prev[i-1][j]*0.25 + gridworld_policy_prev[i+1][j]*0.25 + gridworld_policy_prev[i][j+1]*0.25 + gridworld_policy_prev[i][j]*0.25
     

## final iteration print out
print("Final Iteration: " , str(iteration_policy))
print(gridworld_policy)

print("============================================")
print("================= PART 2 Value Iteration =====================")
print("============================================")


iteration_value = 0

## Initalize a policy pi arbitrarily
## using 5x5 instead of 4x4 like we did inside of class
gridworld_value = np.zeros((5,5))

## using 5x5 instead of 4x4 like we did inside of class
gridworld_value_prev = np.ones((5,5))

while not np.array_equal(gridworld_value, gridworld_value_prev):
	if iteration_value == 0 or iteration_value == 1 or iteration_value == 2:
		print("Iteration: " , str(iteration_value))
		print(gridworld_value)
	iteration_value += 1;

	## pi = pi prime
	gridworld_value_prev = np.copy(gridworld_value)
	for i in range(0,5):
		for j in range(0,5):
			if i == 0:
				if j != 0 and j != 4:
					## edge cant do down
					curr_max =  max(
								-1 + discount * gridworld_value_prev[i][j], 
								-1 + discount * gridworld_value_prev[i+1][j], 
								-1 + discount * gridworld_value_prev[i][j+1],
								-1 + discount * gridworld_value_prev[i][j-1])
					gridworld_value[i][j] = curr_max
				elif j == 4:
					## bounded edge can't do right
					curr_max = max(
								-1 + discount * gridworld_value_prev[i][j],
								-1 + discount * gridworld_value_prev[i+1][j],
								-1 + discount * gridworld_value_prev[i][j],
								-1 + discount *gridworld_value_prev[i][j-1])
					gridworld_value[i][j] =  curr_max
			elif i == 4:
				if j != 0 and j != 4:
					curr_max = max(
								-1 + discount * gridworld_value_prev[i-1][j],
								-1 + 1*gridworld_value_prev[i][j],
								-1 + 1*gridworld_value_prev[i][j+1],
								-1 + 1*gridworld_value_prev[i][j-1])
					gridworld_value[i][j] = curr_max  
				elif j == 0:
					curr_max = 
					gridworld_value[i][j] =  max(-1 + discount * gridworld_value_prev[i-1][j],-1 + 1*gridworld_value_prev[i][j],-1 + 1*gridworld_value_prev[i][j+1],-1 + 1*gridworld_value_prev[i][j])
			else:
				if j != 0 and j != 4:
					curr_max = 
					gridworld_value[i][j] =  max(-1 + discount * gridworld_value_prev[i-1][j],-1 + 1*gridworld_value_prev[i+1][j],-1 + 1*gridworld_value_prev[i][j+1],-1 + 1*gridworld_value_prev[i][j-1])
				elif j == 4:
					curr_max = 
					gridworld_value[i][j] =  max(-1 + discount * gridworld_value_prev[i-1][j],-1 + 1*gridworld_value_prev[i+1][j],-1 + 1*gridworld_value_prev[i][j],-1 + 1*gridworld_value_prev[i][j-1])
				elif j == 0:
					curr_max = 
					gridworld_value[i][j] =  max(-1 + discount * gridworld_value_prev[i-1][j],-1 + 1*gridworld_value_prev[i+1][j],-1 + 1*gridworld_value_prev[i][j+1],-1 + 1*gridworld_value_prev[i][j])
     

## final iteration print out
print("Final Iteration: " , str(iteration_value))
print(gridworld_value)