from sklearn.datasets import fetch_olivetti_faces
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.svm import SVC
from sklearn import metrics


olivetti_faces = fetch_olivetti_faces()

features = olivetti_faces.data
targets = olivetti_faces.target

# print(features)
# print(targets)

x_train, x_test, y_train, y_test = train_test_split(
    features,
    targets,
    test_size=0.25,
    random_state=0,
    stratify=targets,  # stratify = targets -> defining target variables as labels
)


# -----------------------------------------------------------

# pca = PCA()

# # reducing (eigen vectors) principle components from 4096 to 100

# pca.fit(features)

# plt.plot(pca.explained_variance_, linewidth=2)
# plt.xlabel("Components")
# plt.ylabel("Explained Variance")

# plt.show()


# -----------------------------------------------------------


pca = PCA(
    n_components=100, whiten=True
)  # whiten will improve the predictive accuracy of the downstream estimators

pca.fit(x_train)

# no_of_eigenfaces = len(pca.components_)  # eigen faces = eigen vectors
# print(no_of_eigenfaces)

# eigen_faces = pca.components_.reshape(no_of_eigenfaces, 64, 64)
# print(eigen_faces.ndim)


x_train_pca = pca.transform(
    x_train
)  # reduce the dimensionality of x_train to the selected principle components

x_test_pca = pca.transform(x_test)

print(features.shape)  # (400 , 4096)
print(x_train_pca.shape)  # (300 , 100)

models = [
    ("Logistic Regression", LogisticRegression()),
    ("Support Vector Machine", SVC()),
    ("Naive Bayes Classifier", GaussianNB()),
]

for name, model in models:
    classifier_model = model
    classifier_model.fit(x_train_pca, y_train)
    y_predicted = classifier_model.predict(x_test_pca)

    print("Model Name : %s" % name)
    print("Accuracy : %s" % metrics.accuracy_score(y_test, y_predicted))


print("\n\n\n")

# USING CROSS VALIDATION :

x_ = pca.fit_transform(features)
for name, model in models:
    print("Model : ", name)
    k_folds = KFold(n_splits=5, shuffle=True, random_state=0)
    score = cross_val_score(model, x_, targets, cv=k_folds)
    print("Mean of the cross validation score : %s" % np.mean(score))
