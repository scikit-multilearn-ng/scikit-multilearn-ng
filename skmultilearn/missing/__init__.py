"""
The `skmultilearn.missing` module provides classifiers and methods 
for dealing with missing labels in multi-label classification problems.

Currently the following algorithm adaptation classification schemes are available in scikit-multilearn:

+---------------------------------------+----------------------------------------------------------------------------------+
| Classifier                            | Description                                                                      |
+=======================================+==================================================================================+
| :class:`~skmultilearn.missing.SMiLE`  | Semi-supervised multi-label classification using incomplete label information.   |
+---------------------------------------+----------------------------------------------------------------------------------+

"""

from .smile import SMiLE

__all__ = [
    "SMiLE",
]
