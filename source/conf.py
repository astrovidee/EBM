# Configuration file for the Sphinx documentation builder.
#
# Build the site from the repository root with:
#     sphinx-build -b html source docs

import os
import sys

# Make EBM_one_file.py, one folder up, importable for the API reference.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'EBM'
copyright = ('2025, Vidya Venkatesan (Python version). Original MATLAB model '
             'by Cecilia Bitz, after North and Coakley (1979)')
author = 'Vidya Venkatesan'
release = '1.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',   # writes .nojekyll so GitHub Pages serves the site
]

templates_path = []
exclude_patterns = []
language = 'en'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sourcelink = False
autodoc_member_order = 'bysource'
