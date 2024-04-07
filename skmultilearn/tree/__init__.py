"""
The `skmultilearn.problem_transform.tree` submodule provides tree-based multi-label classifiers.

Available Classifier:
+----------------------------+---------------------------------------------------------------------------------+
| Classifier                 | Description                                                                     |
|============================|=================================================================================|
| `PredictiveClusteringTree` | A predictive clustering tree algorithm for multi-label classification.          |
+----------------------------+---------------------------------------------------------------------------------+

Available criteria's:
+-------------------------+---------------------------------------------------------------------------------+
| Criterion               | Description                                                                     |
|=========================|=================================================================================|
| `GiniCriterion`         | Gini impurity criterion.                                                        |
+-------------------------+---------------------------------------------------------------------------------+
| `EntropyCriterion`      | Information gain criterion.                                                     |
+-------------------------+---------------------------------------------------------------------------------+
| `CorrelationCriterion`  | Correlation criterion.                                                          |
+-------------------------+---------------------------------------------------------------------------------+
"""

from .pct import (
    PredictiveClusteringTree,
    GiniCriterion,
    EntropyCriterion,
    CorrelationCriterion,
)

__all__ = [
    "PredictiveClusteringTree",
    "GiniCriterion",
    "EntropyCriterion",
    "CorrelationCriterion",
]
