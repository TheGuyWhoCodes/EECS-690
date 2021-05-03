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

## first, lets import the data from github
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

## copy paste proj 1
X = dataset.values[:, 0:4]	
Y = dataset.values[:, 4]	
X_train_fold, X_validation_fold, Y_train_fold, Y_validation_fold = train_test_split(X, Y, test_size=0.50, random_state=1)	# k = 2, so we use .50 to represent half

## super hacky way of doing this lmao sorry
X_validation_fold_1 = X_train_fold
X_train_fold_1 = X_validation_fold
Y_validation_fold_1 = Y_train_fold
Y_train_fold_1 = Y_validation_fold

## these need to be in integers for the model below
for i in range(len(Y_train_fold)):
    if Y_train_fold[i] == 'Iris-setosa':
        Y_train_fold[i] = 1
    elif Y_train_fold[i] == 'Iris-virginica':
        Y_train_fold[i] = 2
    elif Y_train_fold[i] == 'Iris-versicolor':
        Y_train_fold[i] = 3

for i in range(len(Y_train_fold_1)):
    if Y_train_fold_1[i] == 'Iris-setosa':
        Y_train_fold_1[i] = 1
    elif Y_train_fold_1[i] == 'Iris-virginica':
        Y_train_fold_1[i] = 2
    elif Y_train_fold_1[i] == 'Iris-versicolor':
        Y_train_fold_1[i] = 3

# lets do the basic one
model = LinearRegression()

model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

# i tried using round(), that sucked, had to do this if else statement to do it
predict = []
for num in np.concatenate((fold1, fold2)):
    if num >= 2.5:
        predict.append(3)
    elif num >= 1.5:
        predict.append(2)
    else:
        predict.append(1)

val = []
for num in np.concatenate((Y_validation_fold, Y_validation_fold_1)):
    if num >= 2.5:
        val.append(3)
    elif num >= 1.5:
        val.append(2)
    else:
        val.append(1)

print('========================')
print('1: Linear Regression')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################

## found out this online
model = make_pipeline(PolynomialFeatures(2), Ridge())

## copy paste
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = []
for num in np.concatenate((fold1, fold2)):
    if num >= 2.5:
        predict.append(3)
    elif num >= 1.5:
        predict.append(2)
    else:
        predict.append(1)

print('========================')
print('2: Polynomial of degree 2 regression (LinearRegression)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################

## found out this online
model = make_pipeline(PolynomialFeatures(3), Ridge())

## copy paste
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = []
for num in np.concatenate((fold1, fold2)):
    if num >= 2.5:
        predict.append(3)
    elif num >= 1.5:
        predict.append(2)
    else:
        predict.append(1)

print('========================')
print('3: Polynomial of degree 3 regression (LinearRegression)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################

model = GaussianNB()

## so here we needed to switch from integer back to the string version
## i hate it here
for i in range(len(Y_train_fold)):
    if Y_train_fold[i] == 1:
        Y_train_fold[i] = 'Iris-setosa'
    elif Y_train_fold[i] == 2:
        Y_train_fold[i] = 'Iris-virginica'
    elif Y_train_fold[i] == 3:
        Y_train_fold[i] = 'Iris-versicolor'

for i in range(len(Y_train_fold_1)):
    if Y_train_fold_1[i] == 1:
        Y_train_fold_1[i] = 'Iris-setosa'
    elif Y_train_fold_1[i] == 2:
        Y_train_fold_1[i] = 'Iris-virginica'
    elif Y_train_fold_1[i] == 3:
        Y_train_fold_1[i] = 'Iris-versicolor'


model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))

print('========================')
print('4: Naïve Baysian (NBClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################
## found online
model = KNeighborsClassifier()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))

print('========================')
print('5: kNN (KNeighborsClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################
## found online
model = LinearDiscriminantAnalysis()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))

print('========================')
print('6: LDA (LinearDiscriminantAnalysis)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

##############################################################################################
## found online
model = QuadraticDiscriminantAnalysis()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))

print('========================')
print('7: QDA (QuadraticDiscriminantAnalysis)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

###############################################################################################

model = LinearSVC(max_iter=10000)
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
print('========================')
print('8: SVM (svm.LinearSVC)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

###############################################################################################

model = DecisionTreeClassifier()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
print('========================')
print('9: Decision Tree (DecisionTreeClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

###############################################################################################

model = RandomForestClassifier()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
print('========================')
print('10: Random Forest (RandomForestClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

###############################################################################################

model = ExtraTreesClassifier()
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
print('========================')
print('11: ExtraTrees (ExtraTreesClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

###############################################################################################

model = MLPClassifier(max_iter=1000)
model.fit(X_train_fold, Y_train_fold)
fold1 = model.predict(X_validation_fold)
model.fit(X_train_fold_1, Y_train_fold_1)
fold2 = model.predict(X_validation_fold_1)

predict = np.concatenate((fold1, fold2))
val = np.concatenate((Y_validation_fold, Y_validation_fold_1))
print('========================')
print('12: NN (neural_network.MLPClassifier)')
print('========================')
print('Accuracy:')
print(accuracy_score(val, predict))
print('Confusion Matrix:')
print(confusion_matrix(val, predict))

 
