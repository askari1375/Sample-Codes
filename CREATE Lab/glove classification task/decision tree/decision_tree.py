# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:17:45 2021

@author: Amirhossein
"""


from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from functions import load_data, load_final_data, find_accuracy


X_train, Y_train = load_data("train")
X_val, Y_val = load_data("validation")
X_test, Y_test = load_data("test")


decision_tree = tree.DecisionTreeClassifier(max_depth = 4, class_weight = 'balanced')
decision_tree.fit(X_train, Y_train)

train_acc = find_accuracy(decision_tree, X_train, Y_train)
val_acc = find_accuracy(decision_tree, X_val, Y_val)
test_acc = find_accuracy(decision_tree, X_test, Y_test)



print("train accuracy\t\t:\t{}".format(train_acc))
print("validation accuracy\t:\t{}".format(val_acc))
print("test accuracy\t\t:\t{}".format(test_acc))

X_train_val, Y_train_val = load_final_data()
final_decision_tree = tree.DecisionTreeClassifier(max_depth = 4, class_weight = 'balanced')
final_decision_tree.fit(X_train_val, Y_train_val)

final_train_acc = find_accuracy(final_decision_tree, X_train_val, Y_train_val)
final_test_acc = find_accuracy(final_decision_tree, X_test, Y_test)


print("train accuracy\t\t:\t{}".format(final_train_acc))
print("test accuracy\t\t:\t{}".format(final_test_acc))


metrics.plot_confusion_matrix(final_decision_tree,
                              X_train_val,
                              Y_train_val,
                              normalize = 'true')

metrics.plot_confusion_matrix(final_decision_tree,
                              X_test,
                              Y_test,
                              normalize = 'true')

plt.show()
