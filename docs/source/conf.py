# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
import pathlib

ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
# sys.path.insert(0, str(ROOT_DIR / 'virne'))
sys.path.insert(0, os.path.abspath(str(ROOT_DIR)))
sys.path.insert(0, str(ROOT_DIR / 'virne'))

# sys.path.insert(0, str(ROOT_DIR))


project = 'virne'
copyright = '2023, GeminiLight'
author = 'GeminiLight'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx_design',
    'sphinx_copybutton'
]

myst_enable_extensions = ["colon_fence"]

autosummary_generate = True

napoleon_use_ivar = True
napoleon_use_admonition_for_references = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output



source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}


html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Virne'


html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#4E98C8',
        'color-brand-content': '#67A4BA',
        'sd-color-success': '#5EA69C',
        'sd-color-info': '#76A2DB',
        'sd-color-warning': '#AD677E',
    },
}