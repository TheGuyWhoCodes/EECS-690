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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

## Load iris data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

## now we split into the 50 / 50 split that johnson asked for in the rubric
## only take first 4 (we are predicting class)
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)
## white chochy with almond milk

## now lets use Naive Bayesian classifier 
models = []
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results = (cv_results)
	names = (name)
	# print("Model Accuracy: ", cv_results.mean())
	# print(results)
	# print(names)


model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print("Model Accuracy: ", accuracy_score(Y_validation, predictions))
print("Model Confusion Matrix: ", confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
