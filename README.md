# scikit-multilearn-ng

[![PyPI version](https://badge.fury.io/py/scikit-multilearn-ng.svg)](https://badge.fury.io/py/scikit-multilearn-ng)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

__scikit-multilearn-ng__ is a Python module capable of performing multi-label
learning tasks and is the follow-up to [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn).
It is built on-top of various scientific Python packages
([numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)) and
follows a similar API to that of [scikit-learn](http://scikit-learn.org/).

More documentation:

- __Website:__ [https://scikit-multilearn-ng.github.io/scikit-multilearn-ng/](https://scikit-multilearn-ng.github.io/scikit-multilearn-ng/)


## Features

- __Native Python implementation.__ A native Python implementation for a variety of multi-label classification algorithms. To see the list of all supported classifiers, check this [link](http://scikit.ml/#classifiers).

- __Interface to Meka.__ A Meka wrapper class is implemented for reference purposes and integration. This provides access to all methods available in MEKA, MULAN, and WEKA &mdash; the reference standard in the field.

- __Builds upon giants!__ Team-up with the power of numpy and scikit. You can use scikit-learn's base classifiers as scikit-multilearn's classifiers. In addition, the two packages follow a similar API.

## Installation & Dependencies

To install scikit-multilearn, simply type the following command:

```bash
$ pip install scikit-multilearn-ng
```

This will install the latest release from the Python package index. If you
wish to install the bleeding-edge version, then clone this repository and
run `setup.py`:

```bash
$ git clone https://github.com/scikit-multilearn-ng/scikit-multilearn-ng.git
$ cd scikit-multilearn-ng
$ python setup.py
```

In most cases requirements are installed when you install using `pip install scikit-multilearn-ng` or run `python setup.py install`. There are also optional dependencies `pip install scikit-multilearn-ng[gpl,keras,meka]` installs the GPL-incurring igraph for for igraph library based clusterers, keras for the keras classifiers and requirements for the meka bridge respectively.

To install `openNE`, run:

```bash
pip install 'openne @ git+https://github.com/thunlp/OpenNE.git@master#subdirectory=src'
```

Note that installing the GPL licensed graphtool, for graphtool based clusters, is complicated, and must be done manually, please see: [graphtool install instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions)

## Basic Usage

**Note: You should use the same import statement as previously with scikit-multilearn (`import skmultilearn`), after installation. This allows for quicker switching to this follow-up version.**

Before proceeding to classification,  this library assumes that you have
a dataset with the following matrices:

- `x_train`, `x_test`: training and test feature matrices of size `(n_samples, n_features)`
- `y_train`, `y_test`: training and test label matrices of size `(n_samples, n_labels)`

Suppose we wanted to use a problem-transformation method called Binary
Relevance, which treats each label as a separate single-label classification
problem, to a Support-vector machine (SVM) classifier, we simply perform
the following tasks:

```python
# Import BinaryRelevance from skmultilearn
from skmultilearn.problem_transform import BinaryRelevance

# Import SVC classifier from sklearn
from sklearn.svm import SVC

# Setup the classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False,True])

# Train
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)
```

More examples and use-cases can be seen in the 
[documentation](https://scikit-multilearn-ng.github.io/scikit-multilearn-ng/_static/example_usage.html).

## Contributing

This project is open for contributions. Here are some of the ways for
you to contribute:

- Bug reports/fix
- Features requests
- Use-case demonstrations
- Documentation updates

In case you want to implement your own multi-label classifier, please 
read our [Developer's Guide](http://scikit.ml/api/base.html) to help
you integrate your implementation in our API.

To make a contribution, just fork this repository, push the changes
in your fork, open up an issue, and make a Pull Request!

## Cite

If you used scikit-multilearn-ng in your research or project, please
cite [the original package scikit-multilearn](https://arxiv.org/abs/1702.01460):

```bibtex
@ARTICLE{2017arXiv170201460S,
   author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
   title = "{A scikit-based Python environment for performing multi-label classification}",
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1702.01460},
   year = 2017,
   month = feb
}
```
