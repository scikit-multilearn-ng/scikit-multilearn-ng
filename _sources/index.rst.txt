.. scikit-multilearn-ng documentation master file, created by
   sphinx-quickstart on Sun Jan 21 19:44:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
Welcome to scikit-multilearn-ng's documentation!
================================================

scikit-multilearn-ng is a Python module capable of performing multi-label learning tasks, building on the legacy of scikit-multilearn. It integrates seamlessly with scientific Python packages like numpy and scipy and follows a familiar API akin to scikit-learn.

.. image:: https://img.shields.io/pypi/v/scikit-multilearn-ng.svg
    :target: https://pypi.org/project/scikit-multilearn-ng/
    :alt: PyPI Version

.. image:: https://img.shields.io/github/license/scikit-multilearn-ng/scikit-multilearn-ng.svg
    :target: https://github.com/scikit-multilearn-ng/scikit-multilearn-ng/blob/master/LICENSE
    :alt: License

Features
--------

- **Native Python Implementation**: Variety of multi-label classification algorithms implemented natively in Python. See the `complete list of classifiers <https://link-to-classifiers>`_.

- **Interface to Meka**: Provides access to all methods available in MEKA, MULAN, and WEKA via a Meka wrapper class.

- **Integration with numpy and scikit**: Use scikit-learn's base classifiers and benefit from numpy's computational efficiency.


Installation & Dependencies
---------------------------

To install scikit-multilearn-ng:

.. code-block:: bash

   $ pip install scikit-multilearn-ng

For the latest development version:

.. code-block:: bash

   $ git clone https://github.com/scikit-multilearn-ng/scikit-multilearn-ng.git
   $ cd scikit-multilearn-ng
   $ python setup.py install

Optional dependencies can be installed as follows:

.. code-block:: bash

   $ pip install scikit-multilearn-ng[gpl,keras,meka]

For installing openNE:

.. code-block:: bash

   $ pip install 'openne @ git+https://github.com/thunlp/OpenNE.git@master#subdirectory=src'

Note: Installing GPL licensed graphtool is more complex. Please refer to the `graphtool install instructions <https://git.skewed.de/count0/graph-tool/wikis/installation-instructions>`_.

Basic Usage
-----------

.. code-block:: python

   # Import BinaryRelevance from skmultilearn
   from skmultilearn.problem_transform import BinaryRelevance
   from sklearn.svm import SVC

   # Setup the classifier
   classifier = BinaryRelevance(classifier=SVC(), require_dense=[False, True])

   # Train
   classifier.fit(X_train, y_train)

   # Predict
   predictions = classifier.predict(X_test)

More examples and use cases will be added soon.

Contributing
------------

Contributions to scikit-multilearn-ng are welcome! Here are some ways to contribute:

- Reporting or fixing bugs
- Requesting features
- Demonstrating use-cases
- Updating documentation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
