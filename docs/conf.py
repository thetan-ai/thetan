"""Sphinx configuration for the thetan website.

Pages are MyST markdown (myst-parser), the API reference comes from the
numpydoc docstrings (autosummary + numpydoc) and the gallery from
docs/examples (sphinx-gallery). Everything is pure Python: no Node.js.

Build it with ``spin docs``, which builds thetan first, runs sphinx-build
with the ``dirhtml`` builder and puts the landing page (docs/_landing) at
the root of the site, with these docs under ``docs/``.
"""

import warnings

project = "Thetan"
author = "The Thetan developers"
copyright = "2025, The Thetan developers"

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_landing",
    "_templates",
    "examples",
    "README.md",
    "sg_execution_times.rst",
]

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

# -- API reference ------------------------------------------------------------

autodoc_default_options = {"members": True, "undoc-members": True}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = False

# numpydoc reports a malformed docstring through warnings.warn rather than
# the Sphinx logger, so sphinx-build -W would not see it.
warnings.filterwarnings(
    "error", message=r"(?s).*in the docstring of ", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="Matplotlib is currently using agg", category=UserWarning
)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Gallery ------------------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py$",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_signature": False,
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib",),
}

# -- HTML ---------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Thetan"
html_static_path = ["_static"]
html_css_files = ["thetan.css"]
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "text": "thetan",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "github_url": "https://github.com/thetan-ai/thetan",
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}
html_sidebars = {"index": [], "citing": []}
