import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scikit-multilearn-ng'
copyright = '2014-2016, Piotr Szymański'
author = 'scikit-multilearn-ng'
version = '0.0.6'
release = 'v0.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # ... other extensions ...
    'sphinx.ext.autodoc',
]
sys.path.insert(0, '../skmultilearn/')

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**test**']

rst_epilog = """
-----------------------------------------------------------------------------------------------------------------------------

**Cite us**

If you use scikit-multilearn-ng in your research and publish it, please consider citing scikit-multilearn:

.. code-block:: bibtex

    @ARTICLE{2017arXiv170201460S,
        author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
        title = "{A scikit-based Python environment for performing multi-label classification}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1702.01460},
        primaryClass = "cs.LG",
        keywords = {Computer Science - Learning, Computer Science - Mathematical Software},
        year = 2017,
        month = feb,
    }

"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
    'github_user': 'scikit-multilearn-ng',
    'github_repo': 'scikit-multilearn-ng',
    'github_button': True,
    'github_type': 'star',
    'github_count': True,
    'extra_nav_links': {
        'Index': 'genindex.html',
        'Module Index': 'py-modindex.html',
        'GitHub Repository': 'https://github.com/scikit-multilearn-ng/scikit-multilearn-ng/',
        'License': 'source/license.html',
        'PyPI': 'https://pypi.org/project/scikit-multilearn-ng/',
    }
}
html_static_path = ['_static']
html_short_title = 'scikit-multilearn-ng'
html_show_copyright = False
html_show_sphinx = False
html_logo = "https://avatars.githubusercontent.com/u/153663050?s=200&v=4"
html_meta = {
    'description': 'scikit-multilearn-ng is an advanced Python module for multi-label and multi-target machine learning, extending scikit-learn.',
    'keywords': 'multi-label learning, machine learning, python, scikit-learn, classification, scikit-multilearn, scikit-multilearn-ng, multi-target learning',
}

