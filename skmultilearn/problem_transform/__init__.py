"""
The :mod:`skmultilearn.problem_transform` module provides classifiers
that follow the problem transformation approaches to multi-label classification.

The problem transformation approach to multi-label classification converts multi-label problems to
single-label problems: single-class or multi-class.


+-----------------------------------------------------------------------------+------------------------------------------------+
| Classifier                                                                  | Description                                    |
+=============================================================================+================================================+
| :class:`~skmultilearn.problem_transform.BinaryRelevance`                    |  treats each label as a separate single-class  |
|                                                                             |  classification problem                        |
+-----------------------------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.ClassifierChain`                    |  treats each label as a part of a conditioned  |
|                                                                             |  chain of single-class classification problems |
+-----------------------------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.ClassificationHeterogeneousFeature` | augments the feature set                       |
|                                                                             | with extra features derived from label         |
|                                                                             | probabilities and resolves cyclic dependencies |
|                                                                             | between features and labels iteratively        |
+-----------------------------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.LabelPowerset`                      | treats each label combination as a separate    |
|                                                                             | class with one multi-class classification      |
|                                                                             | problem                                        |
+-----------------------------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.InstanceBasedLogisticRegression`    | combines instance-based learning with logistic |
|                                                                             | regression, using neighbors' information as    |
|                                                                             | features. It has a K-Nearest Neighbor layer    |
|                                                                             | followed by Logistic Regression classifiers.   |
+-----------------------------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.StructuredGridSearchCV`             | performs hyperparameter tuning for each label  |
|                                                                             | classifier, considering BR&CC structural       |
|                                                                             | properties. It searches for optimal classifiers|
|                                                                             | with fine-tuned parameters for each label.     |
+-----------------------------------------------------------------------------+------------------------------------------------+
"""

from .br import BinaryRelevance
from .cc import ClassifierChain
from .chf import ClassificationHeterogeneousFeature
from .gsc import StructuredGridSearchCV
from .lp import LabelPowerset
from .iblr import InstanceBasedLogisticRegression


__all__ = [
    "BinaryRelevance",
    "ClassifierChain",
    "ClassificationHeterogeneousFeature",
    "LabelPowerset",
    "InstanceBasedLogisticRegression",
    "StructuredGridSearchCV",
]
