import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PredictiveClusteringTree(BaseEstimator, ClassifierMixin):
    """
    A predictive clustering tree algorithm for multi-label classification.

    This algorithm builds a decision tree structure where each leaf node represents a multi-label classifier
    trained on a subset of the data. It recursively partitions the feature space based on the variance reduction
    criterion, aiming to find splits that lead to maximal variance reduction in the label space.

    Parameters
    ----------
    base_classifier : estimator, default=DecisionTreeClassifier()
        The base classifier used at each leaf node of the tree.
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.
    tree_ : Node
        The root node of the decision tree.

    Methods
    -------
    fit(X, y)
        Fit the predictive clustering tree to the training data.
    predict(X)
        Predict multi-label outputs for the input data.
    """
    def __init__(self, base_classifier=DecisionTreeClassifier(), max_depth=5, min_samples_split=2, min_samples_leaf=1):
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive.")

        self.base_classifier = base_classifier
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    class Node:
        """
        Inner class representing a node in the predictive clustering tree.

        Attributes
        ----------
        feature_index : int or None
            The index of the feature used for splitting at this node.
        threshold : float or None
            The threshold value for the split.
        left : Node or None
            The left child node.
        right : Node or None
            The right child node.
        classifier : estimator or None
            The classifier associated with the leaf node.
        """
        def __init__(self):
            self.feature_index = None
            self.threshold = None
            self.left = None
            self.right = None
            self.classifier = None

    def fit(self, X, y):
        """
        Fit the predictive clustering tree to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.
        y : array-like of shape (n_samples, n_labels)
            The binary indicator matrix with label assignments.

        Returns
        -------
        self : object
            The fitted instance of the classifier.
        """
        X, y = check_X_y(X, y, multi_output=True, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the predictive clustering tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.
        y : array-like of shape (n_samples, n_labels)
            The binary indicator matrix with label assignments.
        depth : int, optional
            The current depth of the tree (default is 0).

        Returns
        -------
        node : Node
            The root node of the subtree.
        """
        if len(np.unique(y, axis=0)) == 1 or depth >= self.max_depth or len(X) < self.min_samples_split:
            node = self.Node()
            node.classifier = clone(self.base_classifier).fit(X, y)
            return node

        best_idx, best_thr = None, None
        max_variance_reduction = -np.inf

        for idx in range(self.n_features_in_):
            thresholds = np.unique(X[:, idx])[1:-1]
            
            for thr in thresholds:
                left_idx = X[:, idx] < thr
                right_idx = ~left_idx
                if np.sum(left_idx) >= self.min_samples_leaf and np.sum(right_idx) >= self.min_samples_leaf:
                    y_left, y_right = y[left_idx], y[right_idx]
                    variance_reduction = self._multi_label_variance_reduction(y, y_left, y_right)
                    if variance_reduction > max_variance_reduction:
                        best_idx, best_thr, max_variance_reduction = idx, thr, variance_reduction

        if best_idx is not None:
            left_idx = X[:, best_idx] < best_thr
            X_left, y_left = X[left_idx], y[left_idx]
            X_right, y_right = X[~left_idx], y[~left_idx]
            node = self.Node()
            node.feature_index = best_idx
            node.threshold = best_thr
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)
            return node
        else:
            node = self.Node()
            node.classifier = clone(self.base_classifier).fit(X, y)
            return node

    def _multi_label_variance_reduction(self, y, y_left, y_right):
        """
        Calculate the multi-label variance reduction.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_labels)
            The original label assignments.
        y_left : array-like of shape (n_samples_left, n_labels)
            The label assignments for the left split.
        y_right : array-like of shape (n_samples_right, n_labels)
            The label assignments for the right split.

        Returns
        -------
        variance_reduction : float
            The variance reduction achieved by the split.
        """
        total_variance = np.mean(np.var(y, axis=0))
        left_variance = np.mean(np.var(y_left, axis=0))
        right_variance = np.mean(np.var(y_right, axis=0))
        weight_left = y_left.shape[0] / y.shape[0]
        weight_right = y_right.shape[0] / y.shape[0]
        variance_reduction = total_variance - (weight_left * left_variance + weight_right * right_variance)
        return variance_reduction

    def predict(self, X):
        """
        Predict multi-label outputs for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        predictions : array-like of shape (n_samples, n_labels)
            The binary indicator matrix with predicted label assignments.
        """
        check_is_fitted(self, ['tree_', 'n_features_in_'])
        X = check_array(X)
        predictions = np.array([self._predict(inputs, self.tree_) for inputs in X])
        return predictions

    def _predict(self, inputs, node):
        """
        Recursively predicts labels for the input data.

        Parameters
        ----------
        inputs : array-like of shape (n_features,)
            The input feature vector.
        node : Node
            The current node in the tree.

        Returns
        -------
        prediction : array-like of shape (n_labels,)
            The predicted label assignments.
        """
        if node.classifier:
            return node.classifier.predict([inputs])[0]
        elif inputs[node.feature_index] < node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
