# Development

## Build and test

Thetan uses [meson-python](https://meson-python.readthedocs.io) and
[spin](https://github.com/scientific-python/spin). The Cython extensions
are never compiled inside the source tree, so build before importing:

```bash
pip install -r requirements/build.txt -r requirements/test.txt
spin build          # compile into build/ and install into build-install/
spin test                            # run pytest against that build
spin test -t thetan.tests.test_dbscan   # one test module
spin test -- --pyargs thetan -k dbscan  # tests matching a pattern
```

Cython runs with `--warning-errors`, so any Cython warning fails the
build. New Python modules or test files must be listed in
`thetan/meson.build` or `thetan/tests/meson.build` to be installed.

## Build the website

The website is a [Sphinx](https://www.sphinx-doc.org) project in `docs/`.
Pages are MyST markdown, read by
[myst-parser](https://myst-parser.readthedocs.io). Every tool is a Python
package; nothing else is needed.

```bash
pip install -r requirements/docs.txt
spin docs                 # build thetan, then the API pages, the gallery and the site
spin docs --serve         # same, then serve it on http://127.0.0.1:8000
spin docs --no-plot       # do not run the gallery examples
spin docs --clean         # start from scratch
```

The site lands in `docs/_build/site`: the landing page (`docs/_landing`)
at its root, the Sphinx docs under `docs/`. Links are relative, so the
folder can be served from any URL path, as on GitHub Pages. Warnings fail
the build.

The API reference
: Each page in `docs/api/` documents one module with `automodule`. The
  docstrings are parsed by [numpydoc](https://numpydoc.readthedocs.io).
  To document a new module, add a page there and list it in
  `docs/api/index.md`.

The gallery
: [sphinx-gallery](https://sphinx-gallery.github.io) runs every script in
  `docs/examples/` and writes its page, output and figures to
  `docs/gallery/`, which is generated: do not edit it by hand. Scripts
  are run again only when they change.

## Write an example

An example is a Python script in `docs/examples/`, in the sphinx-gallery
layout:

```python
"""
Title of the example
====================

One paragraph that describes it. The first paragraph is also the summary
on the gallery card.
"""

import numpy as np

# %%
# A section title
# ---------------
#
# Comment lines at the top of a cell are rendered as reStructuredText:
# ``code``, *emphasis*, lists and links.

x = np.arange(3)
print(x)
```

Cells run in order in one namespace. What they print and the matplotlib
figures they leave open appear under their code, and the first figure is
the thumbnail. Seed the random generators so that the page is the same on
every build.

## Docstrings

Public functions are documented in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.
The API pages render the Parameters, Returns, Raises, Notes and Examples
sections.
