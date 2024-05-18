import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from skmultilearn.tree import (
    GiniCriterion,
    EntropyCriterion,
    CorrelationCriterion,
    PredictiveClusteringTree,
)


class TestSplitCriteria(unittest.TestCase):
    """
    Test cases for the split criteria calculations.
    """

    def setUp(self):
        self.labels = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])

    def test_gini_impurity(self):
        criterion = GiniCriterion()
        impurity = criterion.calculate_impurity(self.labels)
        self.assertIsInstance(impurity, float, "Impurity should be a float")
        self.assertGreaterEqual(impurity, 0, "Impurity should be non-negative")

    def test_entropy_impurity(self):
        criterion = EntropyCriterion()
        impurity = criterion.calculate_impurity(self.labels)
        self.assertIsInstance(impurity, float, "Impurity should be a float")
        self.assertGreaterEqual(impurity, 0, "Impurity should be non-negative")

    def test_correlation_impurity(self):
        criterion = CorrelationCriterion()
        impurity = criterion.calculate_impurity(self.labels)
        self.assertIsInstance(impurity, float, "Impurity should be a float")
        self.assertGreaterEqual(impurity, 0, "Impurity should be non-negative")


class PCCTest(ClassifierBaseTest):
    # TODO: Support sparse matrices
    def test_if_sparse_classification_works(self):
        classifier = PredictiveClusteringTree(classifier=DecisionTreeClassifier())

        self.assertClassifierWorksWithSparsity(classifier, "sparse")

    def test_if_dense_classification_works(self):
        classifier = PredictiveClusteringTree(classifier=DecisionTreeClassifier())

        self.assertClassifierWorksWithSparsity(classifier, "dense")


if __name__ == "__main__":
    unittest.main()
