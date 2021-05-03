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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks

## first, lets import the data from github
url = "imbalanced_iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

## copy paste proj 1
X = dataset.values[:, 0:4]	
Y = dataset.values[:, 4]	
X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(X, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half

kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
model = MLPClassifier(max_iter=1000)
predict = cross_val_predict(model, X, Y, cv=kfold)
print('========================')
print('Part 1: Basic Case')
print('========================')
print('Accuracy:')
print(accuracy_score(Y, predict))
print('Confusion Matrix:')
print(confusion_matrix(Y, predict))
matrix = confusion_matrix(Y, predict);
print('========================')
print('Class Balanced Accuracy')
print('========================')

per1 = matrix[0][0] / (matrix[0][0]+matrix[0][1]+matrix[0][2])
recall1 = matrix[0][0] / (matrix[0][0]+matrix[1][0]+matrix[2][0])
min1 = min(per1, recall1)

per2 = matrix[1][1] / (matrix[1][0]+matrix[1][1]+matrix[1][2])
recall2 = matrix[1][1] / (matrix[0][1]+matrix[1][1]+matrix[2][1])
min2 = min(per2, recall2)

per3 = matrix[2][2] / (matrix[2][0]+matrix[2][1]+matrix[2][2])
recall3 = matrix[2][2] / (matrix[0][2]+matrix[1][2]+matrix[2][2])
min3 = min(per3, recall3)

classBalanced = (min1 + min2 + min3) / 3
print(classBalanced)

print('========================')
print('Balanced Accuracy')
print('========================')
averageRecall = (recall1+recall2+recall3)/3

tn1 = matrix[0][0] + matrix[2][2] + matrix[2][0] + matrix[2][2]
fp1 = matrix[1][0] + matrix[1][2]
spec1 = tn1 / (tn1+fp1)

tn2 = matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix[2][2]
fp2 = matrix[0][1] + matrix[0][2]
spec2 = tn2 / (tn2+fp2)

tn3 = matrix[1][0] + matrix[1][1] + matrix[0][1] + matrix[0][0]
fp3 = matrix[2][0] + matrix[2][1]
spec3 = tn3 / (tn3+fp3)

avg1 = (spec1+recall1)/2
avg2 = (spec2+recall2)/2
avg3 = (spec3+recall3)/2
print("Balanced Accuracy:")
print((avg1+avg2+avg3)/3)
print('========================')
print('SKlearn Balanced Accuracy')
print('========================')

print("Accuracy: ")
print(balanced_accuracy_score(Y, predict))

print('========================')
print('Part 2: Oversampling ')
print('========================')

print('========================')
print('Random Oversampling ')
print('========================')

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))

print('========================')
print('SMOTE Oversampling ')
print('========================')

ros = SMOTE(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))

print('========================')
print('ADASYN Oversampling ')
print('========================')

ros = ADASYN(random_state=0, sampling_strategy='minority')
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))

print('========================')
print('Part 3: Undersampling ')
print('========================')

print('========================')
print('Random Undersampling ')
print('========================')

ros = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))

print('========================')
print('Cluster Undersampling ')
print('========================')

ros = ClusterCentroids(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))

print('========================')
print('Tomek Undersampling ')
print('========================')

ros = TomekLinks()
X_resampled, y_resampled = ros.fit_resample(X, Y)
kfold = StratifiedKFold(n_splits=2)
predictions = cross_val_predict(model, X_resampled, y_resampled)
print("Accuracy from Random Sampler: ")
print(accuracy_score(y_resampled, predictions))

print('Confusion Matrix:')
print(confusion_matrix(y_resampled, predictions))