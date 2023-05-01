from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

# print feature names
print('Features: ', cancer.feature_names)

# print label type of cancer ('mal', 'ben')
print('Labels: ', cancer.target_names)

# print data(feature) shape
print(cancer.data.shape)

print()  # break

# print the cancer dta features (top 5 records)
print(cancer.data[0:5])

# print the target set cancer labels (0: mal,  1: ben)
print(cancer.target)

# Split the dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

print()  # break

clf = svm.SVC(kernel='linear')

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Model Accuracy--how often is the classifier correct?
print('Model Accuracy: ', metrics.accuracy_score(y_test, y_pred))

# Model Precision--how often is the classifier correct?
# Precision is the ratio of true positives to total positive predictions.
# How accurate the model is when it predicts than an instance belongs to a specific class.
# Indicates lower rate of false positives.
print('Precision: ', metrics.precision_score(y_test, y_pred))

# Recall--Higher recall indicates a lower rate of false positives.
print('Recall: ', metrics.recall_score(y_test, y_pred))
