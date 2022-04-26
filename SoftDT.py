from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np


class SoftDT(BaseEstimator):

    def __init__(self):
        # estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator = DecisionTreeClassifier(random_state=0)
        self.estimator = estimator

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # self.classes_ = unique_labels(y)
        return self.estimator.fit(X, y)

    def predict(self, X):
        # D = self.predict_proba(X)
        # return self.classes_[np.argmax(D, axis=1)]
        # return self.estimator.predict(X)
        return np.apply_along_axis(self.predict_proba, 1, X)

    # not sure if needed - work without
    def classes_(self):
        if self.estimator:
            return self.estimator.classes_

    def predict_proba(self, X):
        alpha = 0.9
        n = 5

        children_left = self.estimator.tree_.children_left
        children_right = self.estimator.tree_.children_right
        features = self.estimator.tree_.feature
        thresholds = self.estimator.tree_.threshold
        tree_values = self.estimator.tree_.value

        total_prob = np.zeros(shape=tree_values[0].shape)

        for i in range(n):
            current_node_id = 0

            while not children_left[current_node_id] == children_right[current_node_id]:
                curr_feature = features[current_node_id]
                curr_threshold = thresholds[current_node_id]

                left = X[curr_feature] <= curr_threshold
                random_num = np.random.rand()
                # change original direction
                if random_num > alpha:
                    left = not left

                if left:
                    current_node_id = children_left[current_node_id]
                else:
                    current_node_id = children_right[current_node_id]

            prob = tree_values[current_node_id] / np.sum(tree_values[current_node_id])
            total_prob += prob
        return total_prob

    # def update_node(self, current_node_id):
