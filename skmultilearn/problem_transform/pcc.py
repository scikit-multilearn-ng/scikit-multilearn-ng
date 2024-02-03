import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.base import clone
from sklearn.utils import validation
from sklearn.exceptions import NotFittedError
from ..problem_transform import ClassifierChain


class ProbabilisticClassifierChain(ClassifierChain):
    """
    Probabilistic Classifier Chain for Multi-Label Classification

    This class implements a Probabilistic Classifier Chain (PCC), an extension of 
    Jesse Read's Classifier Chains. It learns a chain of classifiers, each predicting 
    a label conditioned on the input features and the predictions of preceding classifiers 
    in the chain. This approach models joint label distributions to capture label correlations.

    Each classifier in the chain is trained on an augmented input space that includes the 
    original features (X) and the predictions of all previous classifiers in the chain.

    The implementation adapted and changed from:
    https://github.com/ChristianSch/skml/blob/master/skml/problem_transformation/probabilistic_classifier_chain.py

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        A scikit-learn compatible base classifier. This classifier is used as the base model
        for each step in the classifier chain.
    require_dense : [bool, bool], optional
        Indicates whether the base classifier requires dense representations for input features 
        and label matrices in fit/predict. If not provided, it defaults to using sparse 
        representations unless the base classifier is an instance of 
        :class:`~skmultilearn.base.MLClassifierBase`, in which case dense representations are used.
    order : List[int], permutation of ``range(n_labels)``, optional
        the order in which the chain should go through labels, the default is ``range(n_labels)``

    Attributes
    ----------
    classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `n_labels`
        A list of classifiers, one for each label, trained per partition as per the chain order.

    References
    ----------
    If using this implementation, please cite the scikit-multilearn library and the relevant paper:

    .. code-block:: bibtex

        @inproceedings{Dembczynski2010BayesOM,
          title={Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains},
          author={Krzysztof Dembczynski and Weiwei Cheng and Eyke H{\"u}llermeier},
          booktitle={International Conference on Machine Learning},
          year={2010},
          url={https://api.semanticscholar.org/CorpusID:6418797}
        }

    Examples
    --------
    An example of using Probabilistic Classifier Chain with an :class:`sklearn.svm.SVC`:

    .. code-block:: python

        from skmultilearn.problem_transform import ProbabilisticClassifierChain
        from sklearn.svm import SVC

        # Initialize Probabilistic Classifier Chain with an SVM classifier
        classifier = ProbabilisticClassifierChain(
            classifier = SVC(probability=True),
            require_dense = [False, True]
        )

        # Train
        classifier.fit(X_train, y_train)

        # Predict
        predictions = classifier.predict(X_test)

    To optimize the classifier chain with grid search, one can vary both the base classifier
    and its parameters:

    .. code-block:: python

        from skmultilearn.problem_transform import ProbabilisticClassifierChain
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC

        parameters = [
            {
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.7, 1.0],
            },
            {
                'classifier': [SVC(probability=True)],
                'classifier__kernel': ['rbf', 'linear'],
            },
        ]

        clf = GridSearchCV(ProbabilisticClassifierChain(), parameters, scoring='accuracy')
        clf.fit(X, y)

        print(clf.best_params_, clf.best_score_)

    """

    def __init__(self, classifier=None, require_dense=None, order=None):
        super(ProbabilisticClassifierChain, self).__init__(classifier, require_dense)
        self.order = order
        self.copyable_attrs = ["classifier", "require_dense", "order"]

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        validation.check_is_fitted(self, 'classifiers_')
        X_extended = self._ensure_input_format(
            X, sparse_format="csc", enforce_sparse=True
        )

        y_pred = []
        N_instances = X_extended.shape[0]

        for n in range(N_instances):
            x = X_extended[n].reshape(1, -1)
            y_out = None
            p_max = 0

            for b in range(2 ** self._label_count):
                p = np.zeros((1, self._label_count))
                y = np.array(list(map(int, np.binary_repr(b, width=self._label_count))))

                for i, c in enumerate(self.classifiers_):
                    if i == 0:
                        p[0, i] = c.predict_proba(self._ensure_input_format(x))[0][y[i]]
                    else:
                        y_slice = csr_matrix(y[:i].reshape(1, -1))
                        stacked = hstack((x, y_slice)).reshape(1, -1)
                        p[0, i] = c.predict_proba(self._ensure_input_format(stacked))[0][y[i]]

                pp = np.prod(p)

                if pp > p_max:
                    y_out = y
                    p_max = pp

            y_pred.append(y_out)

        return csr_matrix(y_pred)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `float in [0.0, 1.0]`, shape=(n_samples, n_labels)
            matrix with label assignment probabilities
        """
        validation.check_is_fitted(self, 'classifiers_')
        X_extended = self._ensure_input_format(
            X, sparse_format="csc", enforce_sparse=True
        )

        results = []
        N_instances = X_extended.shape[0]

        for n in range(N_instances):
            x = X_extended[n].reshape(1, -1)
            y_out = None
            p_max = 0

            for b in range(2 ** self._label_count):
                p = np.zeros((1, self._label_count))
                y = np.array(list(map(int, np.binary_repr(b, width=self._label_count))))

                for i, c in enumerate(self.classifiers_):
                    if i == 0:
                        p[0, i] = c.predict_proba(self._ensure_input_format(x))[0][y[i]]
                    else:
                        y_slice = csr_matrix(y[:i].reshape(1, -1))
                        stacked = hstack([x, y_slice]).reshape(1, -1)
                        p[0, i] = c.predict_proba(self._ensure_input_format(stacked))[0][y[i]]

                pp = np.prod(p)

                if pp > p_max:
                    y_out = csr_matrix(p)
                    p_max = pp

            results.append(y_out)

        y_proba = vstack(results)
        return y_proba
