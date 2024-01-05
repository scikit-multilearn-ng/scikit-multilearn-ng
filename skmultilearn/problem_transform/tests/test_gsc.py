import unittest
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

from skmultilearn.problem_transform import StructuredGridSearchCV, BinaryRelevance
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class GSCTest(ClassifierBaseTest):
    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        parameters = {
            'classifier': [SVC(probability=True)],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }

        classifier = StructuredGridSearchCV(
            estimator=BinaryRelevance(require_dense=[True, True]),
            param_grid=parameters,
            scoring='accuracy'
        )

        self.assertClassifierWorksWithSparsity(classifier, "sparse")
        self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        parameters = {
            'classifier': [GaussianNB()],
            'var_smoothing': [1e-10, 1],
        }

        classifier = StructuredGridSearchCV(
            estimator=BinaryRelevance(require_dense=[True, True]),
            param_grid=parameters,
            scoring='accuracy'
        )

        self.assertClassifierWorksWithSparsity(classifier, "sparse")
        self.assertClassifierPredictsProbabilities(classifier, "sparse")


if __name__ == "__main__":
    unittest.main()
