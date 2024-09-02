import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from abc import ABC, abstractmethod


class SplitCriterion(ABC):
    @abstractmethod
    def calculate_impurity(self, labels):
        """Calculate the impurity of a dataset."""
        raise NotImplementedError

    @abstractmethod
    def calculate_gain(self, base_impurity, left_labels, right_labels):
        """Calculate the information gain from a split."""
        raise NotImplementedError


class GiniCriterion(SplitCriterion):
    def calculate_impurity(self, labels):
        if labels.size == 0:
            return 0
        if issparse(labels):
            label_sums = labels.sum(axis=0).A1
        else:
            label_sums = np.sum(labels, axis=0)
        total_label_count = np.sum(label_sums)
        if total_label_count == 0:
            return np.inf
        label_probs = label_sums / total_label_count
        return 1 - np.sum(np.multiply(label_probs, label_probs))

    def calculate_gain(self, base_impurity, left_labels, right_labels):
        left_impurity = self.calculate_impurity(left_labels)
        right_impurity = self.calculate_impurity(right_labels)
        weight_left = left_labels.shape[0] / (
            left_labels.shape[0] + right_labels.shape[0]
        )
        weight_right = 1 - weight_left
        return base_impurity - (
            weight_left * left_impurity + weight_right * right_impurity
        )


class EntropyCriterion(SplitCriterion):
    def calculate_impurity(self, labels):
        if issparse(labels):
            if not labels.nnz:
                return 0
            label_sums = labels.sum(axis=0).A1
        else:
            if not labels.any():
                return 0
            label_sums = np.sum(labels, axis=0)
        total = label_sums.sum()
        if total == 0:
            return 0
        label_probs = label_sums / total
        label_probs = label_probs[label_probs > 0]
        return -np.sum(label_probs * np.log2(label_probs))

    def calculate_gain(self, base_impurity, left_labels, right_labels):
        left_entropy = self.calculate_impurity(left_labels)
        right_entropy = self.calculate_impurity(right_labels)
        total = left_labels.shape[0] + right_labels.shape[0]
        weighted_avg_entropy = (
            left_labels.shape[0] * left_entropy + right_labels.shape[0] * right_entropy
        ) / total
        return base_impurity - weighted_avg_entropy


class CorrelationCriterion(SplitCriterion):
    def calculate_impurity(self, labels):
        if labels.size == 0:
            return np.inf
        if issparse(labels):
            labels = labels.toarray()
        if np.all(np.all(labels == labels[0, :], axis=0)):
            return np.inf
        std_labels = np.std(labels, axis=0)
        valid_cols = std_labels > 0
        if np.sum(valid_cols) < 2:
            return np.inf
        valid_labels = labels[:, valid_cols]
        corr_matrix = np.abs(np.corrcoef(valid_labels, rowvar=False))
        np.fill_diagonal(corr_matrix, 0)
        avg_corr = np.mean(corr_matrix)
        return avg_corr

    def calculate_gain(self, base_impurity, left_labels, right_labels):
        left_impurity = self.calculate_impurity(left_labels)
        right_impurity = self.calculate_impurity(right_labels)
        total_samples = left_labels.shape[0] + right_labels.shape[0]
        weighted_avg_impurity = (
            left_labels.shape[0] * left_impurity
            + right_labels.shape[0] * right_impurity
        ) / total_samples
        return base_impurity - weighted_avg_impurity


class PredictiveClusteringTree(BaseEstimator, ClassifierMixin):
    """
    A predictive clustering tree (PCT) algorithm for multi-label classification that supports multiple split criteria.

    This algorithm constructs a decision tree where each leaf node represents a multi-label classifier trained on a
    subset of the data. It partitions the feature space recursively, aiming to find splits that lead to optimal
    separation based on a specified impurity criterion, thereby potentially capturing complex label dependencies more
    effectively.

    The flexibility to choose between different splitting criteria (e.g., Gini, entropy, correlation) allows for tailored
    approaches to handling multi-label data, enabling the algorithm to better accommodate the specific characteristics
    and correlations present in the labels.

    Parameters
    ----------
    classifier : estimator, default=DecisionTreeClassifier()
        The base classifier used at each leaf node of the tree. This classifier is trained on the subsets of data
        determined by the tree splits.¨
    criterion : SplitCriterion instance, default=GiniCriterion()
        The criterion used to evaluate splits. Must be an instance of a class that extends the SplitCriterion
        abstract base class.
    max_depth : int, default=5
        The maximum depth of the tree. Limits the number of recursive splits to prevent overfitting.
    min_samples_split : int, default=2
        The minimum number of samples required to consider splitting an internal node. Helps prevent creating nodes
        with too few samples.
    min_samples_leaf : int, default=1
        The minimum number of samples a leaf node must have. Ensures that each leaf has a minimum size,impacting the
        granularity of the model.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data upon fitting the model.
    tree_ : Node
        The root node of the decision tree. Each node in the tree represents a decision point or a leaf with an
        associated classifier.

    Methods
    -------
    fit(X, y):
        Fit the predictive clustering tree model to the training data.
    predict(X):
        Predict multi-label outputs for the input data using the trained tree.

    Notes
    -----
    .. note ::

        The tree-building process relies heavily on the chosen split criterion's ability to evaluate and select the most
        informative splits. Custom split criteria can be implemented by extending the SplitCriterion abstract base class.

    .. note ::

        Currenlty only dense input data is supported.

    Examples
    --------
    .. code-block:: python

        from sklearn.datasets import make_multilabel_classification
        X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=3, n_labels=2, random_state=42)
        pct = PredictiveClusteringTree(criterion=GiniCriterion(), max_depth=4, min_samples_split=2, min_samples_leaf=1)
        pct.fit(X, y)
        pct.predict(X[0:5])
    """

    def __init__(
        self,
        classifier=DecisionTreeClassifier(),
        criterion=GiniCriterion(),
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.criterion = criterion
        self.classifier = classifier
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

    def _check_all_rows_same(self, y):
        if issparse(y):
            if y.shape[0] <= 1:
                return True
            first_row = y[0].toarray()
            for i in range(1, y.shape[0]):
                if not np.array_equal(first_row, y[i].toarray()):
                    return False
            return True
        else:
            return len(np.unique(y, axis=0)) == 1

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
        if issparse(X):
            X = X.tocsc()
        base_impurity = self.criterion.calculate_impurity(y)
        all_rows_not_same = self._check_all_rows_same(y)
        if (
            all_rows_not_same
            or depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
        ):
            node = self.Node()
            dense_y = y.toarray() if issparse(y) else y
            node.classifier = clone(self.classifier).fit(X, dense_y)
            return node

        best_gain = -np.inf
        best_idx, best_thr = None, None

        for idx in range(self.n_features_in_):
            if issparse(X):
                column_values = X[:, idx].toarray().ravel()
            else:
                column_values = X[:, idx]
            thresholds = np.unique(column_values)
            for thr in thresholds:
                if issparse(X):
                    left_idx = X[:, idx].toarray().ravel() < thr
                    right_idx = np.logical_not(left_idx)
                else:
                    left_idx = column_values < thr
                    right_idx = ~left_idx

                if np.sum(left_idx) >= self.min_samples_leaf and np.sum(right_idx) >= self.min_samples_leaf:
                    y_left, y_right = y[left_idx], y[right_idx]
                    gain = self.criterion.calculate_gain(base_impurity, y_left, y_right)
                    if gain != np.inf and gain > best_gain:
                        best_gain = gain
                        best_idx, best_thr = idx, thr

        if best_idx is not None:
            if issparse(X):
                left_idx = X[:, best_idx].toarray().ravel() < best_thr
                right_idx = np.logical_not(left_idx)
            else:
                left_idx = X[:, best_idx] < best_thr
                right_idx = ~left_idx

            X_left, y_left = X[left_idx], y[left_idx]
            X_right, y_right = X[right_idx], y[right_idx]
            node = self.Node()
            node.feature_index = best_idx
            node.threshold = best_thr
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)
            return node
        else:
            node = self.Node()
            dense_y = y.toarray() if issparse(y) else y
            node.classifier = clone(self.classifier).fit(X, dense_y)
            return node

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
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X, accept_sparse=True)
        if issparse(X):
            predictions = [self._predict(X[i].toarray().ravel(), self.tree_) for i in range(X.shape[0])]
        else:
            predictions = [self._predict(inputs, self.tree_) for inputs in X]

        return np.array(predictions)

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
