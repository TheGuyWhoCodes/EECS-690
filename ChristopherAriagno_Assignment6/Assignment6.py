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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Load iris data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

X = dataset.values[:, 0:4]
Y = []
w = {}
for k in range(1, 21):
	kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
	w[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
for x in range(0, 150):
    Y.append(0)
plt.figure()
plt.plot(list(w.keys()), list(w.values()))
plt.xlabel("K Clusters")
plt.ylabel("Reconstruction Error")
plt.show()

elbowK = 3
# line from the top but with 3 you know the vibeee
model = KMeans(n_clusters=elbowK, max_iter=1000).fit(X)
predictions = model.predict(X)

print("================================")
print("Elbow K Confusion Matrix")
print("================================")
print(np.transpose(confusion_matrix(predictions, Y)))
print("================================")
print("Elbow K Accuracy")
print("================================")
print(accuracy_score(predictions, Y))

print("\n\n================================")
print("Gaussian Mixture Models (GMM)")
print("================================\n\n")

aic = []

for n in range(1,21):
	aicModel = GaussianMixture(n_components = n, covariance_type = "diag").fit(X)
	aic.append(aicModel.aic(X))

plt.figure()
plt.plot(range(1,21), aic)
plt.xlabel("Number of clusters")
plt.ylabel("AIC")
plt.show()

aic_elbow_k = 3 ## found manually lol

predictions = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag').fit_predict(X)

print("================================")
print("AIC Confusion Matrix")
print("================================")
print(np.transpose(confusion_matrix(predictions, Y)))
print("================================")
print("AIC Accuracy")
print("================================")
print(accuracy_score(predictions, Y))


print("\n\n================================")
print("Gaussian Mixture Models (GMM)")
print("================================\n\n")
## ugh bic now

bic = []

for n in range(1,21):
	bicModel = GaussianMixture(n_components = n, covariance_type = "diag").fit(X)
	bic.append(bicModel.bic(X))

plt.figure()
plt.plot(range(1,21), bic)
plt.xlabel("Number of clusters")
plt.ylabel("BIC")
plt.show()

bic_elbow_k = 3 ## found manually lol

predictions = GaussianMixture(n_components=bic_elbow_k, covariance_type='diag').fit_predict(X)

print("================================")
print("BIC Confusion Matrix")
print("================================")
print(np.transpose(confusion_matrix(predictions, Y)))
print("================================")
print("BIC Accuracy")
print("================================")
print(accuracy_score(predictions, Y))