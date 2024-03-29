"""
The `skmultilearn.problem_transform.tree` submodule provides tree-based multi-label classifiers.

Available Classifier:
+----------------------------+---------------------------------------------------------------------------------+
| Classifier                 | Description                                                                     |
|============================|=================================================================================|
| `PredictiveClusteringTree` | A predictive clustering tree algorithm for multi-label classification.          |
+----------------------------+---------------------------------------------------------------------------------+
"""

from .pct import PredictiveClusteringTree

__all__ = ["PredictiveClusteringTree"]
