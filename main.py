import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from SoftDT import SoftDT

# regular
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
# clf.fit(X_train, y_train)
#
# n_nodes = clf.tree_.node_count
# children_left = clf.tree_.children_left
# children_right = clf.tree_.children_right
# feature = clf.tree_.feature
# threshold = clf.tree_.threshold
#
# tree_values = clf.tree_.value
#
#
# tree.plot_tree(clf)
# plt.show()
# print()


soft_clf = SoftDT()
soft_clf.fit(X_train, y_train)

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

# tree.plot_tree(clf.estimator)
# plt.show()
z_soft = soft_clf.predict(X_test)
# a = z_soft.reshape(X_test.shape[0],X_test.shape[2])
z = clf.predict(X_test[0, :].reshape(1,-1))
print()