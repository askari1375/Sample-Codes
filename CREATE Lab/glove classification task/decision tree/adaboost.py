# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:18:01 2021

@author: Amirhossein
"""

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from functions import load_data, load_final_data, find_accuracy


X_train, Y_train = load_data("train")
X_val, Y_val = load_data("validation")
X_test, Y_test = load_data("test")


base_estimator = tree.DecisionTreeClassifier(max_depth = 5, class_weight = 'balanced')
clf = AdaBoostClassifier(base_estimator, n_estimators = 100, random_state = 0)

clf.fit(X_train, Y_train)

train_acc = find_accuracy(clf, X_train, Y_train)
val_acc = find_accuracy(clf, X_val, Y_val)

print("train accuracy\t\t:\t{}".format(train_acc))
print("validation accuracy\t:\t{}".format(val_acc))

X_train_val, Y_train_val = load_final_data()
final_clf = AdaBoostClassifier(base_estimator, n_estimators = 100, random_state = 0)
final_clf.fit(X_train_val, Y_train_val)

final_train_acc = find_accuracy(final_clf, X_train_val, Y_train_val)
final_test_acc = find_accuracy(final_clf, X_test, Y_test)


print("train accuracy\t\t:\t{}".format(final_train_acc))
print("test accuracy\t\t:\t{}".format(final_test_acc))


metrics.plot_confusion_matrix(final_clf,
                              X_train_val,
                              Y_train_val,
                              normalize = 'true')

metrics.plot_confusion_matrix(final_clf,
                              X_test,
                              Y_test,
                              normalize = 'true')

plt.show()