# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EBM'
copyright = '2025, Vidya Venkatesan (python version), Cecilia Bitz coded up the original version from North&Coakley1979 in Matlab'
author = 'Vidya Venkatesan (python version), Cecilia Bitz coded up the original version from North&Coakley1979 in Matlab'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Required Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',           # For Google-style docstrings
    'sphinx.ext.viewcode',           # Link to source code
    'sphinx.ext.autodoc.typehints',  # Type hints in docstrings
    'sphinx.ext.todo',               # Support for TODO directives
    'sphinx.ext.githubpages'         # For GitHub Pages compatibility
]

import os
import sys
sys.path.insert(0, os.path.abspath('/Users/astrovidee/Dropbox/EBM'))




templates_path = ['_templates']
exclude_patterns = []

language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

html_show_sourcelink = True
suppress_warnings = ['autodoc.mock']

